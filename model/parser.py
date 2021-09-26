import sys 
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, os.path.dirname(currentdir)) 
from binarization import *
from pre import *
from tokenizer import *
from modules.encoder import *
from modules.decoder import *
from modules.attention import *
from modules.embeddings import *
from fairseq import options, utils
from fairseq.modules import *
from fairseq.models.transformer import *
from fairseq import tasks,options
from fairseq.data import dictionary
from fairseq.utils import import_user_module
import os
from six.moves import xrange as range
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from fastai.text import *
import nltk
from nltk.tree import Tree
nltk.download('punkt')
from asdl.hypothesis import *
from asdl.transition_system import *
from common.registerable import Registrable
from components.decode_hypothesis import DecodeHypothesis
from components.action_info import *
from components.dataset_new import Batch
from common.utils import update_args, init_arg_parser
from model.pointer_net import PointerNet
from components.vocab import Vocab, VocabEntry

@Registrable.register('default_parser')
class Parser(nn.Module):

    def __init__(self, args,encoder, decoder, vocab, transition_system, src_embed, pad_idx=0):
        super(Parser, self).__init__()
        self.args = args
        self.vocab = vocab
        self.src_vocab=self.vocab
        self.input_dim=args.action_embed_size 
        self.input_dim += args.action_embed_size 
        self.input_dim += args.field_embed_size 
        self.input_dim += args.type_embed_size

        self.transition_system = transition_system
        self.grammar = self.transition_system.grammar

        self.src_embed = src_embed
        self.production_embed = nn.Embedding(len(transition_system.grammar) + 1, args.action_embed_size)
        self.primitive_embed = nn.Embedding(len(vocab.primitive), args.action_embed_size)
        self.field_embed = nn.Embedding(len(transition_system.grammar.fields), args.field_embed_size)
        self.type_embed = nn.Embedding(len(transition_system.grammar.types), args.type_embed_size)

        args.src_embed_size=args.action_embed_size

        self.encoder = encoder
        self.ende_inp=nn.Linear(args.src_embed_size,self.input_dim)

        self.decoder = decoder
        self.out = nn.Linear(self.input_dim, args.action_embed_size)
        self.src_pointer_net = PointerNet(query_vec_size=args.action_embed_size, src_encoding_size=self.input_dim)
        self.primitive_predictor = nn.Linear(args.action_embed_size, 2)
        self.pad_idx = pad_idx
       
        self.query_vec_to_action_embed = nn.Linear(args.action_embed_size, args.embed_size, bias=args.readout == 'non_linear')
               
        self.query_vec_to_primitive_embed = self.query_vec_to_action_embed
        self.query_vec_to_src_embed=self.query_vec_to_action_embed

        self.read_out_act = torch.tanh 
        
        self.production_readout_b = nn.Parameter(torch.FloatTensor(len(transition_system.grammar) + 1).zero_())
        self.tgt_token_readout_b = nn.Parameter(torch.FloatTensor(len(vocab.primitive)).zero_())
        self.production_readout = lambda q: F.linear(self.read_out_act(self.query_vec_to_action_embed(q)),
                                                         self.production_embed.weight)
        self.tgt_token_readout = lambda q: F.linear(self.read_out_act(self.query_vec_to_primitive_embed(q)),
                                                        self.primitive_embed.weight)
        self.tgt_text_readout =lambda q: F.linear(self.read_out_act(self.query_vec_to_src_embed(q)),
                                                        self.src_embed.weight) 

        if args.cuda:
            self.new_long_tensor = torch.cuda.LongTensor
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_long_tensor = torch.LongTensor
            self.new_tensor = torch.FloatTensor
        self.zero_action_embed = Variable(self.new_tensor(args.action_embed_size).zero_())
       
    def step(self, batch,**kwargs):

        out=[]
        for example in batch.examples:
            
            out_row=[]
            x=Variable(self.new_tensor(self.input_dim).zero_(),requires_grad=False)
            offset = self.args.action_embed_size  # prev_action
            offset += self.args.action_embed_size
            offset += self.args.field_embed_size
            
            x[offset: offset + self.args.type_embed_size] = self.type_embed.weight[self.grammar.type2id[self.grammar.root_type]]
            out_row.append(x)
            for t in range(batch.max_action_num-1):
                embeds=[]
                if t<len(example.tgt_actions):
                    try:
                        a_tm1=example.tgt_actions[t].action
                    except:
                        a_tm1=example.tgt_actions[t]
                    if isinstance(a_tm1, ApplyRuleAction):
                        a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[a_tm1.production]]
                    elif isinstance(a_tm1, ReduceAction):
                        a_tm1_embed = self.production_embed.weight[len(self.grammar)]
                    elif isinstance(a_tm1, GenTextAction):
                        a_tm1_embed=self.src_embed.weight[self.src_vocab.source[a_tm1.text]]
                    elif isinstance(a_tm1,LangMask):
                        a_tm1_embed=self.lang_mask_embed.weight[0]
                    elif isinstance(a_tm1, MaskAction):
                        a_tm1_embed=self.mask_embed.weight[0]
                    else:

                        a_tm1_embed = self.primitive_embed.weight[self.vocab.primitive[a_tm1.token]]

                else:
                    a_tm1_embed =  Variable(self.new_tensor(self.args.action_embed_size).zero_())
                embeds.append(a_tm1_embed)
                parent_production_embed = self.production_embed.weight[torch.tensor(batch.frontier_prod_idx(example,t+1))]
                embeds.append(parent_production_embed)
                parent_field_embed = self.field_embed.weight[torch.tensor(batch.frontier_field_idx(example,t+1))]
                embeds.append(parent_field_embed)
                parent_field_type_embed = self.type_embed.weight[torch.tensor(batch.frontier_field_type_idx(example,t+1))]
                embeds.append(parent_field_type_embed)

                embeds=torch.cat(embeds,dim=-1)
                out_row.append(embeds)

            out.append(torch.stack(out_row))
        out=torch.stack(out)
        
        encoder_output = self.encoder(batch.leaves,batch.nodes, batch.spans, **kwargs)
       
        decoder_out = self.decoder(out, encoder_out=encoder_output, **kwargs)
        
        return self.out(decoder_out[0]),encoder_output['leaves']


    def score(self, examples, **kwargs):
        batch = Batch(examples, self.grammar, self.vocab,src_vocab=self.src_vocab, copy=self.args.no_copy is False, cuda=self.args.cuda)
        query_vectors,src_encodings=self.step(batch,**kwargs)
        apply_rule_prob = F.softmax(self.production_readout(query_vectors), dim=-1)
        tgt_apply_rule_prob = torch.gather(apply_rule_prob, dim=2,
                                           index=batch.apply_rule_idx_matrix.unsqueeze(dim=2)).squeeze(2)
 
        gen_from_vocab_prob = F.softmax(self.tgt_token_readout(query_vectors), dim=-1)
        tgt_primitive_gen_from_vocab_prob = torch.gather(gen_from_vocab_prob, dim=2,
                                                         index=batch.primitive_idx_matrix.unsqueeze(dim=2)).squeeze(2)
        
        primitive_predictor = F.softmax(self.primitive_predictor(query_vectors), dim=-1)
        primitive_copy_prob = self.src_pointer_net(src_encodings, None, query_vectors)
        tgt_primitive_copy_prob = torch.sum(primitive_copy_prob * batch.primitive_copy_token_idx_mask, dim=-1)
        action_mask_pad = torch.eq(batch.apply_rule_mask + batch.gen_token_mask + batch.primitive_copy_mask, 0.)
        action_mask = 1. - action_mask_pad.float()
        action_prob = tgt_apply_rule_prob * batch.apply_rule_mask + \
                        primitive_predictor[:, :, 0] * tgt_primitive_gen_from_vocab_prob * batch.gen_token_mask + \
                        primitive_predictor[:, :, 1] * tgt_primitive_copy_prob * batch.primitive_copy_mask
        action_prob.data.masked_fill_(action_mask_pad.data, 1.e-7)
        action_prob = action_prob.log() * action_mask
        
        scores = torch.sum(action_prob, dim=1)
        returns = [scores]
        return returns
     
    def forward(self, examples, **kwargs):
        return self.score(examples, **kwargs)
    
    def parse(self, src_leaves, src_nodes, src_spans,orig_leaves=None,beam_size=30,**kwargs):
        primitive_vocab=self.vocab.primitive
        args=self.args
        T = torch.cuda if self.args.cuda else torch
        a = 0
        hypotheses = [DecodeHypothesis()]
        completed_hypotheses = []
        aggregated_primitive_tokens = OrderedDict()
        src=orig_leaves
        for token_pos, token in enumerate(src):
            aggregated_primitive_tokens.setdefault(token, []).append(token_pos)
        
        re_encoder_output = self.encoder(src_leaves,src_nodes, src_spans, **kwargs)
        re_src_encodings=re_encoder_output['leaves']

        
        out=[]
        with torch.no_grad():
            hyp_scores = Variable(self.new_tensor([0.]))
        while len(completed_hypotheses) < beam_size and a<200 :
            hyp_num=len(hypotheses)
            encoder_output={}
            encoder_output['encoder_out']=re_encoder_output['encoder_out'].permute(1,0,2).expand(hyp_num,re_encoder_output['encoder_out'].permute(1,0,2).shape[1],re_encoder_output['encoder_out'].permute(1,0,2).shape[2]).permute(1,0,2)
            encoder_output['encoder_indices']=re_encoder_output['encoder_indices'].expand(hyp_num,re_encoder_output['encoder_indices'].shape[1],re_encoder_output['encoder_indices'].shape[2])
            if re_encoder_output['encoder_padding_mask'] is not None:
                encoder_output['encoder_padding_mask']=re_encoder_output['encoder_padding_mask'].expand(hyp_num,re_encoder_output['encoder_padding_mask'].shape[1])
            else:
                encoder_output['encoder_padding_mask']=None
            
            if re_encoder_output['node_padding_mask'] is not None:

                encoder_output['node_padding_mask']=re_encoder_output['node_padding_mask'].expand(hyp_num,re_encoder_output['node_padding_mask'].shape[1])
            else: 
                encoder_output['node_padding_mask']=None
            src_encodings=re_src_encodings.expand(hyp_num,re_src_encodings.shape[1],re_src_encodings.shape[2]) 

            if a==0:

                with torch.no_grad():
                    embed=Variable(self.new_tensor(self.input_dim).zero_())
                offset = self.args.action_embed_size  # prev_action
                offset += self.args.action_embed_size
                offset += self.args.field_embed_size
                embed[offset: offset + self.args.type_embed_size] = self.type_embed.weight[self.grammar.type2id[self.grammar.root_type]]
                out.append(embed.unsqueeze(dim=0)) 
                x=torch.stack(out)
            else:

                x=torch.stack(out)
            decoder_out = self.decoder(x, encoder_out=encoder_output, **kwargs)
            out1=self.out(decoder_out[0])
            apply_rule_log_prob = F.log_softmax(self.production_readout(out1[:,-1,:]), dim=-1)
            gen_from_vocab_prob = F.softmax(self.tgt_token_readout(out1[:,-1,:]), dim=-1)
            
            primitive_copy_prob = self.src_pointer_net(src_encodings, None, out1[:,-1,:].unsqueeze(dim=1)).squeeze(dim=1)
            primitive_predictor_prob = F.softmax(self.primitive_predictor(out1[:,-1,:]), dim=-1)
            primitive_prob = primitive_predictor_prob[:, 0].unsqueeze(dim=1) * gen_from_vocab_prob
            
            gentoken_prev_hyp_ids = []
            gentoken_new_hyp_unks = {}
            applyrule_new_hyp_scores = []
            applyrule_new_hyp_prod_ids = []
            applyrule_new_hyp_ids = []
            for hyp_id,hyp in enumerate(hypotheses):
                action_types = self.transition_system.get_valid_continuation_types(hyp)
                for action_type in action_types:
                    if action_type == ApplyRuleAction: 
                        
                        productions = self.transition_system.get_valid_continuating_productions(hyp)
                        for production in productions:
                            prod_id = self.grammar.prod2id[production]
                            prod_score = apply_rule_log_prob[hyp_id, prod_id].data.item()
                            new_hyp_score1 = hyp.score+prod_score

                            applyrule_new_hyp_scores.append(new_hyp_score1)
                            applyrule_new_hyp_prod_ids.append(prod_id)
                            applyrule_new_hyp_ids.append(hyp_id)
                    elif action_type == ReduceAction:
                        assert apply_rule_log_prob.shape[-1]==len(self.grammar)+1
                        action_score = apply_rule_log_prob[hyp_id, -1].data.item()
                        new_hyp_score1 = hyp.score+action_score

                        applyrule_new_hyp_scores.append(new_hyp_score1)
                        applyrule_new_hyp_prod_ids.append(len(self.grammar))
                        applyrule_new_hyp_ids.append(hyp_id)
                    else:
                        gentoken_prev_hyp_ids.append(hyp_id)
                        hyp_copy_info = dict()  # of (token_pos, copy_prob)
                        hyp_unk_copy_info = []

                        for token, token_pos_list in aggregated_primitive_tokens.items():
                            sum_copy_prob = torch.gather(primitive_copy_prob[hyp_id], 0, Variable(T.LongTensor(token_pos_list))).sum()
                            gated_copy_prob = primitive_predictor_prob[hyp_id, 1].squeeze() * sum_copy_prob

                            if token in primitive_vocab:
                                token_id = primitive_vocab[token]
                                primitive_prob[hyp_id, token_id] = primitive_prob[hyp_id, token_id] + gated_copy_prob

                                hyp_copy_info[token] = (token_pos_list, gated_copy_prob.data.item())
                            else:
                                hyp_unk_copy_info.append({'token': token, 'token_pos_list': token_pos_list,
                                                            'copy_prob': gated_copy_prob.data.item()})

                        if len(hyp_unk_copy_info) > 0:
                            unk_i = np.array([x['copy_prob'] for x in hyp_unk_copy_info]).argmax()
                            token = hyp_unk_copy_info[unk_i]['token']
                            primitive_prob[hyp_id, primitive_vocab.unk_id] = hyp_unk_copy_info[unk_i]['copy_prob']
                            gentoken_new_hyp_unks[hyp_id]=token

                            hyp_copy_info[token] = (hyp_unk_copy_info[unk_i]['token_pos_list'], hyp_unk_copy_info[unk_i]['copy_prob'])

            new_hyp_scores = None
            if applyrule_new_hyp_scores:
                new_hyp_scores = Variable(self.new_tensor(applyrule_new_hyp_scores))
            if gentoken_prev_hyp_ids:
                primitive_log_prob = torch.log(primitive_prob)
                gen_token_new_hyp_scores = (hyp_scores[gentoken_prev_hyp_ids].unsqueeze(1) + primitive_log_prob[gentoken_prev_hyp_ids, :]).view(-1)

                if new_hyp_scores is None: new_hyp_scores = gen_token_new_hyp_scores
                else: new_hyp_scores = torch.cat([new_hyp_scores, gen_token_new_hyp_scores],dim=-1)
            top_new_hyp_scores, top_new_hyp_pos = torch.topk(new_hyp_scores,
                                                             k=min(beam_size-len(completed_hypotheses),new_hyp_scores.size(0)))
            live_hyp_ids = []
            new_hypotheses = []
            for new_hyp_score_i, new_hyp_pos in zip(top_new_hyp_scores.data.cpu(), top_new_hyp_pos.data.cpu()):
                action_info = ActionInfo()
                if new_hyp_pos.item() < len(applyrule_new_hyp_scores):
                    new_hyp_id = applyrule_new_hyp_ids[new_hyp_pos]
                    new_hyp = hypotheses[new_hyp_id]
                    prod_id = applyrule_new_hyp_prod_ids[new_hyp_pos]
                    if prod_id < len(self.grammar):
                        production = self.grammar.id2prod[prod_id]
                        action = ApplyRuleAction(production)
                    else:
                        action = ReduceAction()
                else:
                    token_id = (new_hyp_pos - len(applyrule_new_hyp_scores)) % primitive_prob.size(-1)
                    k = (new_hyp_pos - len(applyrule_new_hyp_scores)) // primitive_prob.size(-1)
                    new_hyp_id = gentoken_prev_hyp_ids[k]
                    new_hyp = hypotheses[new_hyp_id]
                    if token_id == primitive_vocab.unk_id:
                        try:
                            token = gentoken_new_hyp_unks[k.item()]
                        except:
                            token = primitive_vocab.id2word[primitive_vocab.unk_id]
                    else:
                        token = primitive_vocab.id2word[token_id.item()]

                    action = GenTokenAction(token)
                    if token in aggregated_primitive_tokens:
                            action_info.copy_from_src = True
                            action_info.src_token_position = aggregated_primitive_tokens[token]
                action_info.action = action
                action_info.t = a
                
                if a > 0:
                    action_info.parent_t = new_hyp.frontier_node.created_time
                    action_info.frontier_prod = new_hyp.frontier_node.production
                    action_info.frontier_field =new_hyp.frontier_field.field
                new_hyp =new_hyp.clone_and_apply_action_info(action_info)
                new_hyp.score = new_hyp_score_i
                
                if new_hyp.completed:
                    new_hyp.score =new_hyp.score/(a+1)
                    completed_hypotheses.append(new_hyp)
                else:
                    new_hypotheses.append(new_hyp)
                    live_hyp_ids.append(new_hyp_id)
            if live_hyp_ids:
                out_new=[]
                hypotheses=new_hypotheses
                for i,id in enumerate(live_hyp_ids):
                    out_i=out[id]
                    a_tm1 = hypotheses[i].actions[-1]
                    hypothesis=hypotheses[i]
                
                    embeds=[]
                    if isinstance(a_tm1, ApplyRuleAction):
                        a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[a_tm1.production]]
                            #one dimensional
                    elif isinstance(a_tm1, ReduceAction):
                        a_tm1_embed = self.production_embed.weight[len(self.grammar)]
                    elif isinstance(a_tm1, GenTextAction):
            
                        a_tm1_embed=self.src_embed.weight[self.src_vocab.source[a_tm1.text]]
                    elif isinstance(a_tm1,TreeTextAction):
                        if a_tm1.text in self.tags2id:
                            a_tm1_embed=self.tags_embed.weight[self.tags2id[a_tm1.text]]
                        else: 
                            a_tm1_embed=self.tags_embed.weight[self.tags2id['unk']]
                    else:
                        a_tm1_embed = self.primitive_embed.weight[self.vocab.primitive[a_tm1.token]]
                    embeds.append(a_tm1_embed)
                    frontier_prod_embed = self.production_embed.weight[torch.tensor(\
                        self.grammar.prod2id[hypothesis.frontier_node.production])]
                    embeds.append(frontier_prod_embed)
                    frontier_field_embed = self.field_embed.weight[torch.tensor(\
                        self.grammar.field2id[hypothesis.frontier_field.field])]
                    embeds.append(frontier_field_embed)
                    frontier_field_type = self.type_embed.weight[torch.tensor(\
                        self.grammar.type2id[hypothesis.frontier_field.type])]
                    embeds.append(frontier_field_type)

                    embeds=torch.cat(embeds,dim=-1).unsqueeze(dim=0)
                    out_i=torch.cat((out_i,embeds),dim=0)
                    out_new.append(out_i)
                out=out_new
                hyp_scores = Variable(self.new_tensor([hyp.score for hyp in hypotheses]))
                a+=1
                if a==200:
                    completed_hypotheses.sort(key=lambda hyp: -hyp.score)
                    return completed_hypotheses
            else:
                break

        completed_hypotheses.sort(key=lambda hyp: -hyp.score)
        return completed_hypotheses

    @staticmethod
    def add_parser(parser):
        TransformerModel.add_parser(parser)
        parser.add_argument(
            '--encoder-token-positional-embeddings', default=False, action='store_true',
            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--use_pos', default=False, action='store_true')
    @classmethod 
    def build_model(cls, args,vocab, transition_system, task=None):
        base_architecture(args)
        import_user_module(args)
        args.left_pad_source=False
        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024
        src_dict=Dictionary.load('data/conala/final_vocab.txt')
        src_vocab=VocabEntry()
        for i in range(4, len(src_dict)):
            src_vocab.add(src_dict[i])
        vocab.source=src_vocab
        tgt_dict=Dictionary()
        for i in range(4, len(vocab.primitive)):
            tgt_dict.add_symbol(vocab.primitive.id2word[i],n=1)
        vocab.primitive.change_pad()

        def build_embedding(dictionary, embed_dim,path=None):
            num_embeddings=len(dictionary)
            padding_idx=1

            emb=Embedding(num_embeddings,embed_dim, padding_idx)
            return emb 
        
        encoder_embed_tokens=build_embedding(src_dict, args.encoder_embed_dim, args.encoder_embed_path)
        decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )
        args.encoder_type = getattr(args, 'encoder_type', Encoder)
        args.decoder_type = getattr(args, 'decoder_type', Decoder)
        encoder = args.encoder_type(args, src_dict, encoder_embed_tokens, left_pad=args.left_pad_source)
        decoder = args.decoder_type(args, tgt_dict,decoder_embed_tokens)
        return cls(args, encoder, decoder, vocab, transition_system,src_embed=encoder_embed_tokens)
    def save(self, path):
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        params = {
            'args': self.args,
            'transition_system': self.transition_system,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)
 
    @classmethod
    def load(cls, model_path, cuda=False):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        vocab = params['vocab']
        transition_system = params['transition_system']
        saved_args = params['args']
        update_args(saved_args, init_arg_parser())
        saved_state = params['state_dict']
        saved_args.cuda = cuda

        parser = Parser.build_model(args=saved_args,vocab=vocab,transition_system=transition_system)

        parser.load_state_dict(saved_state)

        if cuda: parser = parser.cuda()
        parser.eval()

        return parser

        
    


