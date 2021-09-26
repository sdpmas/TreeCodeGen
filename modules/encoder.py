# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=wildcard-import
# pylint: disable=W0614
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from fairseq import * 
from fairseq.modules import (
    AdaptiveInput, AdaptiveSoftmax, CharacterTokenEmbedder, LearnedPositionalEmbedding, MultiheadAttention,
    SinusoidalPositionalEmbedding
)
from fairseq.models import (
    FairseqIncrementalDecoder, FairseqEncoder, FairseqLanguageModel, FairseqModel, register_model,
    register_model_architecture,
)
from fairseq.models.transformer import *
from modules.attention import * 
from modules.embeddings import *
import tensorflow as tf
# from modules.nstack_merge_tree_attention import *



class EncoderLayer(nn.Module):
    def __init__(self, args, padding_idx=1, compute_nodes=True, mask_default=False):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.dptree_class = args.dptree_class
        self.padding_idx = padding_idx
        self.mask_default = mask_default
        att_kwargs = {}
        # nstack_mask_fn
        if mask_default:
            att_kwargs['nstack_mask_fn'] = "default"

        self.self_attn = self.dptree_class(
            args, self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout,
            padding_idx=self.padding_idx, **att_kwargs
        )
        
        self.dropout = args.dropout
        print("the encoder dropout is ", args.dropout)
        self.relu_dropout = getattr(args, 'relu_dropout', 0)
        self.input_dropout = getattr(args, 'input_dropout', 0)

        self.relu_dropoute_layer = nn.Dropout(self.relu_dropout)
        self.plain_dropoute_layer = nn.Dropout(self.dropout)
        self.input_dropout_layer = nn.Dropout(self.input_dropout)

        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(2)])


    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x

    
    def forward(self, x_le, x_no, ntree_mask, hier_embed, pad_mask, key_pad, node_pad):
        # x_le:             [n, b, c]
        # x_no:             [m, b, c]
        # ntree_mask:       [bh, n, m, 1]
        # hier_embed:       [bh, n, m, d] c=d*h
        # pad_mask:         [b, 1, n + m, n + m]


        """n/m=no of nodes/leaves
            b=batch_size
            c=embeddings dimension"""
        n, b, dim = x_le.size()
        m, b_, dim_ = x_no.size()

        leaves = x_le
        nodes = x_no

        x = torch.cat((leaves, nodes), 0)
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x = self.input_dropout_layer(x)
        query = x
        k_le = x[:n]
        k_no = x[n:]

        x, weights = self.self_attn(
            query=query,
            key=k_le, value=k_le,
            node_key=k_no, node_value=k_no,
            ntree_mask=ntree_mask, hier_embed=hier_embed,
            pad_mask=pad_mask, key_pad=key_pad, node_pad=node_pad,
            force_self_att=True
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = self.relu_dropoute_layer(x)
        x = self.fc2(x)
        x = self.plain_dropoute_layer(x)

        """alright here is a residual connection, so you can just assume it is the original matrix + something else that helps."""
        x = residual + x

        # assert not torch.isnan(x).any(), f'after maybe_layer_norm problem'
        x = self.maybe_layer_norm(1, x, after=True)
        # assert not torch.isnan(x).any(), f'after maybe_layer_norm problem 2'

        o_leaves = x[:n]
        o_nodes = x[n:]

        return o_leaves, o_nodes, weights

#

class Encoder(FairseqEncoder):
    def __init__(self, args, dictionary, embed_tokens, left_pad=False):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.args = args

        self.gpu_idx = None
        self.model_parallel = False

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions
        # TODO: set params
        # assert args.encoder_embed_dim == args.decoder_embed_dim, f'encoder-decoder dim not work !='
        # assert args.encoder_attention_heads == args.decoder_attention_heads, f'decoder_att_heads !='
        assert not left_pad
        self.heads = args.encoder_attention_heads
        self.encoder_embed_dim = args.encoder_embed_dim
        self.head_dim = self.encoder_embed_dim // self.heads

        self.embed_path = args.encoder_embed_path

        self.embed_path_exists = self.embed_path is not None and os.path.exists(self.embed_path)
        self.embed_pretrained_no_scale = getattr(args, 'embed_pretrained_no_scale', False)
        self.first_layer_nonodes = getattr(args, 'first_layer_nonodes', False)
        self.pretrained_linear = getattr(args, 'pretrained_linear', False)
        self.use_pos = getattr(args, 'use_pos', False)

        self.vanilla_layers = getattr(args, 'vanilla_layers', 0)

        self.attention_rerun = getattr(args, 'attention_rerun', 1)
        assert self.attention_rerun >= 1
        # self.head_dim = self.encoder_embed_dim // self.head_dim

        self.embed_scale = math.sqrt(embed_dim)
        self.leave_embed_scale = 1.0 if self.embed_path_exists and self.embed_pretrained_no_scale else self.embed_scale
        self.node_embed_scale = self.embed_scale
        print(f'leave_embed_scale={self.leave_embed_scale}, node_embed_scale={self.node_embed_scale}')

        self.dptree_class = args.dptree_class

        self.node_embed_init = getattr(args, 'node_embed_init', 'embed')

        self.nstack_hier_embed = getattr(args, 'nstack_hier_embed', False)
        self.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
        self.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
        self.nstack_hier_embed_share = getattr(args, 'nstack_hier_embed_share', False)

        self.nstack_mask_fname = getattr(args, 'nstack_mask_fn', 'default')
        self.nstack_mask_df_layer = getattr(args, 'nstack_mask_df_layer', None)
        if self.nstack_mask_df_layer is None:
            self.nstack_mask_df_layer = [False] * args.encoder_layers
        else:
            assert isinstance(self.nstack_mask_df_layer, (list, tuple))
            assert len(self.nstack_mask_df_layer) == args.encoder_layers, f'{len(self.nstack_mask_df_layer)}'
        self.mutual_level = getattr(args, 'mutual_ancestor_level', 5)
        self.nstack_mask_building_func = MergeWeightMask.acquire_mask_building_fn(
            self.nstack_mask_fname, self.mutual_level)
        self.nstack_default_mask_building_func = MergeWeightMask.acquire_mask_building_fn(
            'default', self.mutual_level)
        self.is_mask_default = self.nstack_mask_fname == 'default' or self.nstack_mask_fname == 'all_all'

        # ---------------- modules --------------------------
        self.embed_dropout_layer = nn.Dropout(self.dropout)
        self.embed_tokens = embed_tokens
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            left_pad=left_pad,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.hier_pos_positions = MergeHierarchicalEmbedding(
            args, args.encoder_layers, self.head_dim, self.heads,
            self.nstack_hier_embed_max_horiz, self.nstack_hier_embed_max_ver,
            self.nstack_hier_embed_share
        ) if self.nstack_hier_embed else None

        self.pretrained_proj = Linear(
            self.encoder_embed_dim, self.encoder_embed_dim) if self.pretrained_linear else None

        if self.vanilla_layers > 0:
            self.vanilla_leave_layers = nn.ModuleList([])
            self.vanilla_leave_layers.extend([
                TransformerEncoderLayer(args)
                for i in range(self.vanilla_layers)
            ])
        else:
            self.vanilla_leave_layers = None

        self.layers = nn.ModuleList([])


        """these are the encoder layers. and this main class is just the representation of a list of encoder layers. everything gets passed into these encoder layers"""
        self.layers.extend([
            EncoderLayer(
                args, padding_idx=self.padding_idx,
                compute_nodes=(i > 0) or not self.first_layer_nonodes,
                mask_default=self.nstack_mask_df_layer[i]
            )
            for i in range(args.encoder_layers)
        ])

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def setup_cuda(self, gpu_idx):
        print(f'[{self.__class__.__name__}] Setup gpu_idx: {gpu_idx}')
        self.gpu_idx = gpu_idx
        self.model_parallel = True

        first_gpu = [
            'embed_dropout_layer', 'embed_tokens', 'embed_positions', 'hier_pos_positions',
            'pretrained_proj', 'vanilla_leave_layers'
        ]
        last_gpu = ['layer_norm']
        for name, module in self.named_children():
            # module._apply(fn)
            if name in first_gpu:
                print(f'|| [0][{name}]: {module}')
                module.cuda(0)
            elif name in last_gpu:
                print(f'|| [{self.gpu_idx[-1]}][{name}]: {module}')
                module.cuda(self.gpu_idx[-1])
            else:
                assert name == 'layers'
                for i, layer in enumerate(self.layers):
                    print(f'|| [{self.gpu_idx[i]}][{name}]: {layer}')
                    layer.cuda(self.gpu_idx[i])

        for k, param in self._parameters.items():
            if param is not None:
                # Tensors stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.

                # param.data = fn(param.data)
                # if param._grad is not None:
                #     param._grad.data = fn(param._grad.data)
                raise NotImplementedError(f'param is not None [{k}]: {param}')

        for key, buf in self._buffers.items():
            if buf is not None:
                print(f'setup cuda for buf: {key}: {buf}')
                self._buffers[key] = buf.cuda(self.gpu_idx[-1])
                # raise NotImplementedError(f'buf is not None [{key}]: {buf}')
        return self

    # def cuda(self, device=None):
    #     return super().cuda(device)
    
    def extra_repr(self):
        return f'rerun={self.attention_rerun},use_pos={self.use_pos}'

    def embed(self, flat_src_tokens, **kwargs):
        assert not isinstance(self.embed_tokens, PhraseAveragePretrainedEmbedding)
        embeddings = self.embed_tokens(flat_src_tokens)
        return embeddings

    def embed_nodes(self, leave_embed, flat_node_tokens, **kwargs):
        if self.node_embed_init == 'embed':
            if self.args.pretrain_embed_mode == 'bert':
                return self.embed_tokens(flat_node_tokens, only_embedding=True)
            return self.embed(flat_node_tokens)
        elif self.node_embed_init == 'zero':
            b, l = flat_node_tokens.size()
            return torch.zeros(b, l, leave_embed.size(-1), dtype=leave_embed.dtype, device=leave_embed.device)
        else:
            raise ValueError(f'{self.node_embed_init} ???')

    @property
    def recusive_params(self):
        params = list(itertools.chain(*[x.recusive_params for x in self.layers]))
        return params

    def model_parallel_forward(self, src_node_leaves, src_node_nodes, src_node_indices, **kwargs):
        src_label_leaves = kwargs.get('src_label_leaves', None)
        b, n = src_node_leaves.size()
        b_, m = src_node_nodes.size()
        h = self.heads
        assert b == b_, f'{src_node_leaves.size()} != {src_node_nodes.size()}'
        leave_embeddings = self.embed(src_node_leaves)
        if self.use_pos:
            leave_embeddings += self.embed(src_label_leaves)
        node_embeddings = self.embed_nodes(leave_embeddings, src_node_nodes)

        leave_x = self.leave_embed_scale * leave_embeddings
        node_x = self.node_embed_scale * node_embeddings

        if self.pretrained_linear:
            leave_x = self.pretrained_proj(leave_x)

        if self.embed_positions is not None:
            leave_x += self.embed_positions(src_node_leaves)

        leave_x = self.embed_dropout_layer(leave_x)
        node_x = self.embed_dropout_layer(node_x)

        leave_x = leave_x.transpose(0, 1)
        node_x = node_x.transpose(0, 1)

        spans = src_node_indices

        key_pad = src_node_leaves.eq(self.padding_idx)
        node_pad = src_node_nodes.eq(self.padding_idx)
        if not key_pad.any() and not node_pad.any():
            key_pad = node_pad = None

        # build preliminaries
        device = leave_x.device
        ntree_mask = MergeStackNodesOnAffinityValueAttention.get_ntree_mask(n, spans, self.heads)
        pad_mask = self.nstack_mask_building_func(device, h, key_pad, node_pad, spans, b, n + m, n, m, **kwargs)
        default_pad_mask = pad_mask if self.is_mask_default else self.nstack_default_mask_building_func(
            device, h, key_pad, node_pad, spans, b, n + m, n, m, **kwargs
        )
        hier_embeds = self.hier_pos_positions(n, spans) if self.nstack_hier_embed else [None] * len(self.layers)

        if self.vanilla_layers > 0:
            leave = leave_x
            for layer in self.vanilla_leave_layers:
                leave = layer(leave, key_pad)
            leave_x = leave

        attention_dict = {}
        assert self.attention_rerun == 1, f'self.attention_rerun = {self.attention_rerun}'
        for i, (layer, hier_embed, is_maskdf) in enumerate(zip(self.layers, hier_embeds, self.nstack_mask_df_layer)):
            pmask = default_pad_mask if is_maskdf else pad_mask
            try:
                if i > 0:
                    leave_x = leave_x.cuda(i)
                    node_x = node_x.cuda(i)
                    ntree_mask = ntree_mask.cuda(i)
                    hier_embed = hier_embed.cuda(i)
                    pmask = pmask.cuda(i)

                    key_pad = key_pad.cuda(i) if key_pad is not None else key_pad
                    node_pad = node_pad.cuda(i) if node_pad is not None else node_pad
                # print(f'iteration :{i}')
                leave_x, node_x, weights = layer(
                    leave_x, node_x, ntree_mask, hier_embed, pmask, key_pad, node_pad
                )
            except AssertionError as ae:
                print(f'Assert error at layer [{i}]')
                # src_node_leaves, src_node_nodes, src_node_indices
                print(f'sizes: {src_node_leaves.size()} / {src_node_nodes.size()} // {src_node_indices.size()}')
                torch.set_printoptions(profile="full")
                print(src_node_nodes)
                print(f'------------------------------')
                print(src_node_indices)
                print(f'------------------------------')
                torch.set_printoptions(profile="default")
                raise ae
            attention_dict[f'att_{i}'] = weights

        x = torch.cat((leave_x, node_x), 0)
        x = x.cuda(self.gpu_idx[-1])
        if self.normalize:
            x = self.layer_norm(x)

        out_dict = {
            'encoder_out': x,  # (n + m) x b x C
            'encoder_indices': src_node_indices,  # B x m x 2
            'encoder_padding_mask': key_pad,  # B x n
            'node_padding_mask': node_pad,  # B x m
        }
        for k, v in attention_dict.items():
            out_dict[k] = v
        return out_dict

    def forward(self, src_node_leaves, src_node_nodes, src_node_indices,**kwargs):
        """

        :param src_node_leaves:     [b, n]
        :param src_node_nodes:      [b, m]
        :param src_node_indices:    [b, m, 2]
        :param kwargs:
        :return:
        """
        if self.model_parallel:
            return self.model_parallel_forward(src_node_leaves, src_node_nodes, src_node_indices, **kwargs)
        # assert (src_node_leaves < self.embed_tokens.num_embeddings).all(), f'src_tokens >= :max= {src_node_leaves.max()}'
        # assert (src_node_nodes < self.embed_tokens.num_embeddings).all(), f'src_tokens >= :max= {src_node_nodes.max()}'
        src_label_leaves = kwargs.get('src_label_leaves', None)
        # print("the leaves are ",src_node_leaves)
        b, n = src_node_leaves.size()
        b_, m = src_node_nodes.size()
        h = self.heads
        assert b == b_, f'{src_node_leaves.size()} != {src_node_nodes.size()}'
        # leave_embeddings = self.embed(src_node_leaves)
        leave_embeddings = self.embed(src_node_leaves)
        if self.use_pos:
            leave_embeddings += self.embed(src_label_leaves)
        # node_embeddings = self.embed_nodes(leave_embeddings, src_node_nodes)
        node_embeddings = self.embed_nodes(leave_embeddings, src_node_nodes)

        # length=leave_embeddings.shape[1]
        # channels=leave_embeddings.shape[2]
        # pos = torch.from_numpy(self.get_timing_signal_1d(length,channels).numpy()).cuda()
        # leave_embeddings+=pos

        leave_x = self.leave_embed_scale * leave_embeddings
        
        node_x = self.node_embed_scale * node_embeddings
        # leave_x =leave_embeddings
        # node_x =node_embeddings

        if self.pretrained_linear:
            leave_x = self.pretrained_proj(leave_x)

        if self.embed_positions is not None:
            # print("we get here")
            leave_x += self.embed_positions(src_node_leaves)

        leave_x = self.embed_dropout_layer(leave_x)
        node_x = self.embed_dropout_layer(node_x)

        leave_x = leave_x.transpose(0, 1)
        node_x = node_x.transpose(0, 1)

        spans = src_node_indices
        #TODO: sort out the padding inx
        key_pad = src_node_leaves.eq(self.padding_idx)
        node_pad = src_node_nodes.eq(self.padding_idx)
        if not key_pad.any() and not node_pad.any():
            key_pad = node_pad = None

        # build preliminaries
        device = leave_x.device
        ntree_mask = MergeStackNodesOnAffinityValueAttention.get_ntree_mask(n, spans, self.heads)
        # print("encoder done",ntree_mask)
        pad_mask = self.nstack_mask_building_func(device, h, key_pad, node_pad, spans, b, n + m, n, m, **kwargs)
        default_pad_mask = pad_mask if self.is_mask_default else self.nstack_default_mask_building_func(
            device, h, key_pad, node_pad, spans, b, n + m, n, m, **kwargs
        )
        
        hier_embeds = self.hier_pos_positions(n, spans) if self.nstack_hier_embed else [None] * len(self.layers)
        # hier_embeds=[None] * len(self.layers)
        
        if self.vanilla_layers > 0:
            leave = leave_x
            for layer in self.vanilla_leave_layers:
                leave = layer(leave, key_pad)
            leave_x = leave

        attention_dict = {}
        for j in range(self.attention_rerun):
            for i, (layer, hier_embed, is_maskdf) in enumerate(zip(
                    self.layers, hier_embeds, self.nstack_mask_df_layer)):
                pmask = default_pad_mask if is_maskdf else pad_mask
                try:
                    leave_x, node_x, weights = layer(
                        leave_x, node_x, ntree_mask, hier_embed, pmask, key_pad, node_pad
                    )
                except AssertionError as ae:
                    print(f'Assert error at layer [{j}][{i}]')
                    # src_node_leaves, src_node_nodes, src_node_indices
                    print(f'sizes: {src_node_leaves.size()} / {src_node_nodes.size()} // {src_node_indices.size()}')
                    torch.set_printoptions(profile="full")
                    print(src_node_nodes)
                    print(f'------------------------------')
                    print(src_node_indices)
                    print(f'------------------------------')
                    torch.set_printoptions(profile="default")
                    raise ae
                attention_dict[f'att_{i}'] = weights

        x = torch.cat((leave_x, node_x), 0)
        if self.normalize:
            x = self.layer_norm(x)
            leave_x=self.layer_norm(leave_x)

        out_dict = {
            'encoder_out': x,  # (n + m) x b x C
            'encoder_indices': src_node_indices,  # B x m x 2
            'encoder_padding_mask': key_pad,  # B x n
            'node_padding_mask': node_pad, 
            'leaves':leave_x.permute(1,0,2)# B x m
        }
        for k, v in attention_dict.items():
            out_dict[k] = v
        return out_dict

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = encoder_out['encoder_padding_mask'].index_select(0, new_order)
        if encoder_out['node_padding_mask'] is not None:
            encoder_out['node_padding_mask'] = encoder_out['node_padding_mask'].index_select(0, new_order)
        if encoder_out['encoder_indices'] is not None:
            encoder_out['encoder_indices'] = encoder_out['encoder_indices'].index_select(0, new_order)
        for k in encoder_out.keys():
            if "att_" in k:
                encoder_out[k] = encoder_out[k].index_select(0, new_order)

        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)
        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict

