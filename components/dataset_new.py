# coding=utf-8
from collections import OrderedDict

import torch
import numpy as np
from copy import deepcopy
try:
    import cPickle as pickle
except:
    import pickle

from torch.autograd import Variable

from asdl.transition_system import *
from common.utils import cached_property

from model import nn_utils


class Dataset(object):
    def __init__(self, examples):
        self.examples = examples

    @property
    def all_source(self):
        return [e.src_sent for e in self.examples]

    @property
    def all_targets(self):
        return [e.tgt_code for e in self.examples]

    @staticmethod
    def from_bin_file(file_path):
        examples = pickle.load(open(file_path, 'rb'))
        return Dataset(examples)
    
    @staticmethod
    def from_bin_file_dev(file_path):
        examples = pickle.load(open(file_path, 'rb'))
        np.random.shuffle(examples)
        return Dataset(examples)

    def batch_iter(self, batch_size, shuffle=False):
        index_arr = np.arange(len(self.examples))
        if shuffle:
            np.random.shuffle(index_arr)

        batch_num = int(np.ceil(len(self.examples) / float(batch_size)))
        for batch_id in range(batch_num):
            batch_ids = index_arr[batch_size * batch_id: batch_size * (batch_id + 1)]
            batch_examples = [self.examples[i] for i in batch_ids]
            batch_examples.sort(key=lambda e: -len(e.input_actions))

            yield batch_examples

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)


class Example(object):
    def __init__(self, tgt_actions=None,input_actions=None, idx=0, src_sent=None, leaves_nodes=None, tgt_code=None, tgt_ast=None, meta=None,actions=None):
        self.src_sent = src_sent
        self.tgt_code = tgt_code
        self.tgt_ast = tgt_ast
        self.tgt_actions = tgt_actions
        self.leaves_nodes=leaves_nodes  
        self.idx = idx
        self.meta = meta
        self.actions=actions
        self.input_actions=input_actions
        

class Batch(object):
    def __init__(self, examples, grammar, vocab,src_vocab=None ,copy=True, cuda=False):
        self.examples = examples
        self.max_action_num = max(len(e.tgt_actions) for e in self.examples)
        self.vocab = vocab
        self.cuda = cuda
        T = torch.cuda if self.cuda else torch
        self.src_sents = [[i.text for i in e.input_actions] for e in self.examples]
        self.src_sents_len = [len(e.input_actions) for e in self.examples]
        self.input_actions=[e.input_actions for e in self.examples]
        self.input_actions_len=[len(e.input_actions) for e in self.examples]
        
        self.leaves_init=[e.leaves_nodes['leaves'] for e in self.examples]
        

        self.nodes_init=[e.leaves_nodes['nodes'] for e in self.examples]
        self.spans_init=[e.leaves_nodes['spans'] for e in self.examples]
        self.leaves=self.get_leaves(self.leaves_init)
        self.nodes=self.get_nodes(self.nodes_init)
        self.spans=self.get_spans(self.spans_init)
        
        self.leaves_max=max(i.shape[0] for i in self.leaves_init)
        
        self.vocab_src=src_vocab
        self.grammar = grammar
        
        self.copy = copy
        

        self.init_index_tensors()

    def __len__(self):
        return len(self.examples)

    def get_leaves(self, list):
        new_list=[]
        maximum=max(i.shape[0] for i in list)
        for e in list:
            new_e=e.clone()
            for i in range(maximum):
                if i<e.shape[0]:
                    
                    continue
                else:
                    
                    new_e=torch.cat((new_e,torch.tensor([1]).int()), dim=0)
            new_list.append(new_e)
           
        if self.cuda:
            return torch.stack(new_list).long().cuda()
        else:
            return torch.stack(new_list).long()

    
    def get_nodes(self, list):
        new_list=[]
        maximum=max(i.shape[0] for i in list)
        for e in list:
            new_e=e.clone()
            for i in range(maximum):
                if i<e.shape[0]:
                    # new_list.append(new_e)
                    continue
                else:
                   
                    new_e=torch.cat((new_e,torch.tensor([1]).int()), dim=0)
                    # new_list.append(new_e)
            new_list.append(new_e)
            # print("the orig node  and new node is ",e, new_e)

        if self.cuda:
            return torch.stack(new_list).long().cuda()
        else:
            return torch.stack(new_list).long()
    
    def get_spans(self, list):
        new_list=[]
        maximum=max(i.shape[0] for i in list)
        for e in list:
            new_e=e.clone()
            for i in range(maximum):
                if i<e.shape[0]:
                    # new_list.append(new_e)
                    continue
                else:
                    
                    new_e=torch.cat((new_e,torch.tensor([[0,0]]).int()),dim=0)
                    # new_list.append(new_e)
            new_list.append(new_e)
            # print("the orig span  and new span is ",e, new_e)

        if self.cuda:
            return torch.stack(new_list).cuda()
        else:
            return torch.stack(new_list)

    def get_frontier_field_idx(self, t):
        ids = []
        for e in self.examples:
            if t < len(e.tgt_actions):
                ids.append(self.grammar.field2id[e.tgt_actions[t].frontier_field])
            else:
                ids.append(0)

        return Variable(torch.cuda.LongTensor(ids)) if self.cuda else Variable(torch.LongTensor(ids))
    def frontier_field_idx(self, e,t):
       
        if t < len(e.tgt_actions):
            id=self.grammar.field2id[e.tgt_actions[t].frontier_field]
        else:
            id=0
        
        return id

    def frontier_prod_idx(self, e,t):
        
        if t < len(e.tgt_actions):
            id=self.grammar.prod2id[e.tgt_actions[t].frontier_prod]
        else:
            id=0
        
        return id
    def frontier_field_type_idx(self, e,t):
        
        if t < len(e.tgt_actions):
            id=self.grammar.type2id[e.tgt_actions[t].frontier_field.type]
        else:
            id=0

        return id
    

    def get_frontier_prod_idx(self, t):
        ids = []
        for e in self.examples:
            if t < len(e.tgt_actions):
                ids.append(self.grammar.prod2id[e.tgt_actions[t].frontier_prod])
            else:
                ids.append(0)

        return Variable(torch.cuda.LongTensor(ids)) if self.cuda else Variable(torch.LongTensor(ids))

    def get_frontier_field_type_idx(self, t):
        ids = []
        for e in self.examples:
            if t < len(e.tgt_actions):
                ids.append(self.grammar.type2id[e.tgt_actions[t].frontier_field.type])
            else:
                ids.append(0)

        return Variable(torch.cuda.LongTensor(ids)) if self.cuda else Variable(torch.LongTensor(ids))

    def init_index_tensors(self):
        self.apply_rule_idx_matrix = []
        self.apply_rule_mask = []
        self.primitive_idx_matrix = []
        self.gen_token_mask = []
        self.primitive_copy_mask = []
        self.primitive_copy_token_idx_mask = np.zeros((len(self),self.max_action_num,  self.leaves_max), dtype='float32')
        self.text_idx_matrix=[]
        self.gen_text_mask=[]

        for e_id, e in enumerate(self.examples):
        
            app_rule_idx_row = []
            app_rule_mask_row = []
            token_row = []
            gen_token_mask_row = []
            copy_mask_row = []
            gen_text_row=[]
            gen_text_mask_row=[]

            for t in range(self.max_action_num):
                app_rule_idx = app_rule_mask = token_idx = gen_token_mask = copy_mask =gen_text_mask=text_idx= 0
                if t < len(e.tgt_actions):
                    try:
                        action = e.tgt_actions[t].action
                    except:
                        action = e.tgt_actions[t]


                    if isinstance(action, ApplyRuleAction):
                        app_rule_idx = self.grammar.prod2id[action.production]
                        app_rule_mask = 1
                    elif isinstance(action, ReduceAction):
                        app_rule_idx = len(self.grammar)
                        app_rule_mask = 1
                    elif isinstance(action,GenTextAction):
                        # print ('not expected at all')
                        text_idx = self.vocab_src.source[action.text]
                        gen_text_mask=1
                    else:
                        src_sent=self.examples[e_id].leaves_nodes['orig_leaves']
                        token = str(action.token)
                        token_idx = self.vocab.primitive[action.token]

                        token_can_copy = False

                        if self.copy and token in src_sent:
                            token_pos_list = [idx for idx, _token in enumerate(src_sent) if _token == token]
                            self.primitive_copy_token_idx_mask[ e_id,t, token_pos_list] = 1.
                            copy_mask = 1
                            token_can_copy = True

                        if token_can_copy is False or token_idx != self.vocab.primitive.unk_id:
                            gen_token_mask = 1


                app_rule_idx_row.append(app_rule_idx)
                app_rule_mask_row.append(app_rule_mask)

                token_row.append(token_idx)
                gen_token_mask_row.append(gen_token_mask)
                copy_mask_row.append(copy_mask)
                gen_text_row.append(text_idx)
                gen_text_mask_row.append(gen_text_mask)


            self.apply_rule_idx_matrix.append(app_rule_idx_row)
            self.apply_rule_mask.append(app_rule_mask_row)

            self.primitive_idx_matrix.append(token_row)
            self.gen_token_mask.append(gen_token_mask_row)

            self.primitive_copy_mask.append(copy_mask_row)
            self.text_idx_matrix.append(gen_text_row)
            self.gen_text_mask.append(gen_text_mask_row)

        T = torch.cuda if self.cuda else torch
        
        self.apply_rule_idx_matrix = T.LongTensor(self.apply_rule_idx_matrix)
        self.apply_rule_mask = T.FloatTensor(self.apply_rule_mask)
        self.primitive_idx_matrix = T.LongTensor(self.primitive_idx_matrix)
        self.gen_token_mask = T.FloatTensor(self.gen_token_mask)
        self.primitive_copy_mask = T.FloatTensor(self.primitive_copy_mask)
        self.primitive_copy_token_idx_mask = torch.from_numpy(self.primitive_copy_token_idx_mask)
        self.text_idx_matrix = T.LongTensor(self.text_idx_matrix)
        self.gen_text_mask = T.FloatTensor(self.gen_text_mask)
        if self.cuda: self.primitive_copy_token_idx_mask = self.primitive_copy_token_idx_mask.cuda()

    @property
    def primitive_mask(self):
        return 1. - torch.eq(self.gen_token_mask + self.primitive_copy_mask, 0).float()

    @cached_property
    def src_sents_var(self):
        return nn_utils.to_input_variable(self.src_sents, self.vocab.source,
                                          cuda=self.cuda)

    @cached_property
    def src_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.src_sents_len,
                                                    cuda=self.cuda)

    @cached_property
    def token_pos_list(self):
        # (batch_size, src_token_pos, unique_src_token_num)

        for e_id, e in enumerate(self.examples):
            aggregated_primitive_tokens = OrderedDict()
            for token_pos, token in enumerate(e.src_sent):
                aggregated_primitive_tokens.setdefault(token, []).append(token_pos)


