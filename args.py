import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options
from fairseq import utils

from fairseq.modules import (
    AdaptiveInput, AdaptiveSoftmax, CharacterTokenEmbedder, LearnedPositionalEmbedding, MultiheadAttention,
    SinusoidalPositionalEmbedding
)

from fairseq.models import (
    FairseqIncrementalDecoder, FairseqEncoder, FairseqLanguageModel, FairseqModel, register_model, FairseqEncoderModel,
    register_model_architecture,
)
from modules.encoder import *
from modules.decoder import *
from modules.attention import *
from modules.embeddings import *
from model import *
from modules.nstack_tree_attention import *


def nstack2seq_base(args):
    nstack_class_base(args)

def nstack_class_base(args):
    args.encoder_type = getattr(args, 'encoder_type', Encoder)
    args.dptree_class = getattr(args, 'dptree_class', NodeStackOnValueAttention)
    args.placeholder_const = getattr(args, 'placeholder_const', False)
    args.pretrain_embed_mode = getattr(args, 'pretrain_embed_mode', 'const')
    args.on_seq = getattr(args, 'on_seq', 'key')
    args.divide_src_len = getattr(args, 'divide_src_len', True)

    args.src_len_norm = getattr(args, 'src_len_norm', 'none')
    args.nstack_pos_embed = getattr(args, 'nstack_pos_embed', False)
    args.nstack_pos_embed_learned = getattr(args, 'nstack_pos_embed_learned', False)
    args.cum_node = getattr(args, 'cum_node', 'sum')
    args.nstack_linear = getattr(args, 'nstack_linear', False)

    args.wnstack_include_leaves = getattr(args, 'wnstack_include_leaves', True)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'none')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'none')

    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', 'default')
    args.nstack_mask_df_layer = getattr(args, 'nstack_mask_df_layer', None)

    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', False)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    args.nstack_hier_embed_share = getattr(args, 'nstack_hier_embed_share', False)

    args.take_full_dim = getattr(args, 'take_full_dim', False)
    args.hier_embed_right = getattr(args, 'hier_embed_right', False)

    args.dwstack_proj_act = getattr(args, 'dwstack_proj_act', 'none')
    args.node_embed_init = getattr(args, 'node_embed_init', 'embed')

    args.embed_pretrained_no_scale = getattr(args, 'embed_pretrained_no_scale', False)

    args.first_layer_nonodes = getattr(args, 'first_layer_nonodes', False)

    args.vanilla_layers = getattr(args, 'vanilla_layers', 0)

    args.transition_act = getattr(args, 'transition_act', 'none')
    args.transition_dropout = getattr(args, 'transition_dropout', 0.0)

    args.mutual_ancestor_level = getattr(args, 'mutual_ancestor_level', 5)
    args.sep_dwstack_proj_act = getattr(args, 'sep_dwstack_proj_act', 'tanh')

    args.nstack_cross = getattr(args, 'nstack_cross', True)
    #TODO: turn it back on 
    # args.nstack_cross=False
    args.input_dropout = getattr(args, 'input_dropout', 0)
    print(base_architecture)
    base_architecture(args)

#TODO: change the dimensions
def add_iwslt(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 640)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 640)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)


def dwnstack_merge2seq_node_iwslt_onvalue_base_upmean_mean_mlesubenc_allcross_hier(args):
    args.encoder_type = getattr(args, 'encoder_type', Encoder)
    args.decoder_type = getattr(args, 'decoder_type', Decoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    add_iwslt(args)
    nstack2seq_base(args)
   

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', default=0, type=int)
    args = parser.parse_args()
    dwnstack_merge2seq_node_iwslt_onvalue_base_upmean_mean_mlesubenc_allcross_hier(args)






