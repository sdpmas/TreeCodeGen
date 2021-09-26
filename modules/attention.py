import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from fairseq import utils
import os
import functools
from fairseq.models import transformer
from fairseq.modules.multihead_attention import *
from .nstack_merge_tree_attention import *


class MergeStackNodesOnAffinityValueAttention(nn.Module):
    """
    Hierarchical embeddings is embedded from outside
    Encoder Layer computes the embeddings,
        if not share: chunk into multiple embeddings
        pass each to each attention layer!
    """
    WNSTACK_NORM = ['none', 'mean', 'sqrt_mean']
    def __init__(
            self, args, embed_dim, num_heads, dropout=0., bias=True,
            add_bias_kv=False, add_zero_attn=False, padding_idx=1, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.bias = bias
        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn
        self.onnx_trace = False
        self.padding_idx = padding_idx

        self.src_len_norm = getattr(args, 'src_len_norm', 'none')
        self.cum_node = getattr(args, 'cum_node', 'sum')
        self.nstack_on = getattr(args, 'nstack_on', 'value')
        self.nstack_pos_embed = getattr(args, 'nstack_pos_embed', False)
        self.nstack_pos_embed_learned = getattr(args, 'nstack_pos_embed_learned', False)
        self.nstack_linear = getattr(args, 'nstack_linear', False)

        self.wnstack_include_leaves = getattr(args, 'wnstack_include_leaves', True)

        self.wnstack_norm = kwargs.get('wnstack_norm', getattr(args, 'wnstack_norm', 'none'))
        self.wnstack_up_norm = kwargs.get('wnstack_up_norm', getattr(args, 'wnstack_up_norm', 'none'))
        self.nstack_mask_fname = kwargs.get('nstack_mask_fn', getattr(args, 'nstack_mask_fn', 'default'))

        self.mutual_level = getattr(args, 'mutual_ancestor_level', 5)

        self.nstack_mask_func = MergeWeightMask.acquire_masking_fn(self.nstack_mask_fname, self.mutual_level)

        self.nstack_hier_embed = getattr(args, 'nstack_hier_embed', False)
        self.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
        self.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
        self.nstack_hier_embed_share = getattr(args, 'nstack_hier_embed_share', False)

        # max_horiz=100, max_ver=1024

        print(f'Acquire Mask function[{self.nstack_mask_fname}]: {self.nstack_mask_func}')
        assert self.wnstack_norm in self.WNSTACK_NORM
        assert self.nstack_on in ['value', 'key'], f'{self.nstack_on}'

        # if self.nstack_hier_embed:
        self._build_layer()

    @property
    def value_tofloat_mle(self):
        return self.nstack_mask_fname == MergeWeightMask.LEAVES_SUBTREE and value_tofloat_mle

    def extra_repr(self):
        return 'sln={},on={},posemb={},posemb_l={},hier_emb={},cumnode={},linear={},wfname={},upnorm={},hier_share={},dwstack_proj_act={},tofloatmle'.format(
            self.src_len_norm, self.nstack_on, self.nstack_pos_embed, self.nstack_pos_embed_learned,
            self.nstack_hier_embed, self.cum_node,
            self.nstack_linear, self.nstack_mask_fname,
            self.wnstack_up_norm, self.nstack_hier_embed_share, self.dwstack_proj_act, self.value_tofloat_mle
        )

    def _build_layer(self):
        self.src_len_norm = getattr(self.args, 'src_len_norm', 'sqrt')
        self.dwstack_proj_act = getattr(self.args, 'dwstack_proj_act', 'none')

        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * self.embed_dim, self.embed_dim))
        if self.bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * self.embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.bias)

        if self.add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, self.embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, self.embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = self.add_zero_attn

        self.nstack_linear_layer = NstackLinear(self.head_dim, self.head_dim, False) if self.nstack_linear else None

        self.dwstack_linear = transformer.Linear(self.embed_dim, self.num_heads)
        self.project_dwstack_key = lambda x: self.dwstack_linear(x)
        if self.dwstack_proj_act == 'sigmoid':
            self.project_dwstack_key = lambda x: self.dwstack_linear(x).sigmoid()
        elif self.dwstack_proj_act == 'tanh':
            self.project_dwstack_key = lambda x: self.dwstack_linear(x).tanh()

        assert not self.nstack_pos_embed, f'not now'
        self.embed_positions = PositionalEmbedding(
            self.args.max_source_positions, self.head_dim, self.padding_idx,
            left_pad=False,
            learned=self.nstack_pos_embed_learned,
        ) if self.nstack_pos_embed else None

        self.reset_parameters()

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(
            self,
            incremental_state,
            'attn_state',
        ) or {}

    def _set_input_buffer(self, incremental_state, buffer):
        utils.set_incremental_state(
            self,
            incremental_state,
            'attn_state',
            buffer,
        )

    def prepare_dptree_qkv(
            self, query, key, value, node_key, node_value, key_padding_mask=None, node_padding_mask=None,
            incremental_state=None, need_weights=True, static_kv=False,
            compute_query_nodes=True, compute_key_nodes=True, force_self_att=False):
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()
        node_kv_same = node_key.data_ptr() == node_value.data_ptr()

        tgt_len, query_bsz, embed_dim = query.size()
        leave_len, key_bsz, embed_dim_ = key.size()
        node_len, key_bsz, embed_dim_ = node_key.size()

        n = leave_len
        m = node_len
        t = tgt_len
        bsz = b = key_bsz
        assert embed_dim == self.embed_dim
        assert key.size() == value.size()
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                if static_kv:
                    assert kv_same and not qkv_same
                    key = value = node_key = node_value = None
        else:
            saved_state = None

        assert compute_query_nodes
        assert compute_key_nodes

        if qkv_same or force_self_att:
            # self-attention
            q = self.in_proj_q(query)
            assert node_kv_same
            k, v = self.in_proj_kv(key)
            node_k, node_v = self.in_proj_kv(node_key)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = node_k = node_v = None
            else:
                k, v = self.in_proj_kv(key)
                assert node_kv_same
                node_k, node_v = self.in_proj_kv(node_key)
        else:
            raise NotImplementedError

        q *= self.scaling

        assert self.bias_v is None
        assert self.bias_k is None
        assert not self.add_zero_attn

        q = q.contiguous().view(t, query_bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, key_bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, key_bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if node_k is not None:
            node_k = node_k.contiguous().view(-1, key_bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if node_v is not None:
            node_v = node_v.contiguous().view(-1, key_bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            assert static_kv, f'static_kv={static_kv}, only cross-attention impl here'
            def maybe_concat_state_kv(name, x):
                if name in saved_state:
                    prev = saved_state[name]
                    prev = prev.view(b * self.num_heads, -1, self.head_dim)
                    # o = prev if static_kv else torch.cat((prev, x), dim=1)
                    o = prev
                else:
                    o = x
                saved_state[name] = o.view(b, self.num_heads, -1, self.head_dim)
                return o

            k = maybe_concat_state_kv('prev_key', k)
            v = maybe_concat_state_kv('prev_value', v)
            node_k = maybe_concat_state_kv('prev_node_key', node_k)
            node_v = maybe_concat_state_kv('prev_node_value', node_v)

            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(1)
        node_src_len = node_k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(-1) == src_len
        if node_padding_mask is not None:
            assert node_padding_mask.size(-1) == node_src_len

        return (q, k, v, node_k, node_v, key_padding_mask, node_padding_mask, saved_state, src_len, node_src_len, tgt_len, query_bsz)

    @classmethod
    def get_ntree_mask(cls, n, spans, nheads):
        # spans:            [b, m, 2]
        with torch.no_grad():
            b, m, _ = spans.size()
            try:
                rg = torch.arange(0, n, device=spans.device, dtype=spans.dtype).view(1, n, 1, 1)
            except:
                print("the stats are n is ",n)
            spans = spans.unsqueeze(1)
            """important the encodr decoder comes here"""
            mask = (rg >= spans[:, :, :, :1]) ^ (rg > spans[:, :, :, 1:])
            mask = mask.view(b, 1, n, m, 1).contiguous().expand(b, nheads, n, m, 1)
            
            mask = mask.contiguous().view(b * nheads, n, m, 1)
        return mask

    def accumulate_upward(self, leaves, nodes, mask, **kwargs):
        # leaves:           [b, n, 1, c]
        # nodes:            [b, 1, m, c]
        # mask:             [b, n, m, 1]
        float_mask = mask.type_as(nodes)
        node_stack = nodes * float_mask
        if self.wnstack_include_leaves:
            stack = torch.cat([leaves, node_stack], 2)
            upward_cum = torch.cumsum(stack, dim=2)
            node_states = upward_cum[:, :, 1:]
        else:
            stack = node_stack
            upward_cum = torch.cumsum(stack, dim=2)
            node_states = upward_cum
        if self.wnstack_up_norm == 'mean':
            node_states /= torch.cumsum(float_mask, dim=2).clamp_(1.0, 1e4)
        elif self.wnstack_up_norm == 'sqrt_mean':
            node_states /= torch.cumsum(float_mask, dim=2).sqrt_().clamp_(1.0, 1e4)
        node_states *= float_mask
        return node_states

    def accumulate_rightward_backup(self, node_states, mask, right_weight, **kwargs):
        # node_states:      [b, n, m, c]
        # mask:             [b, n, m, 1]

        # node_states_t:    [b, c, m, n]
        # right_weight:     [b, 1, n, 1]
        # rv_node_out:      [b, m, c]
        node_states_t = node_states.transpose(1, 3)
        assert not torch.isnan(right_weight).any(), f'right_weight::right_weight problem: {right_weight}'
        rv_node_out = torch.matmul(node_states_t, right_weight).squeeze_(-1).transpose(1, 2)
        rv_node_out = rv_node_out.clamp_(-1e4, 1e4)
        assert not torch.isnan(rv_node_out).any(), f'rv_node_out::after matmul problem: {rv_node_out}'
        if self.wnstack_norm == 'mean':
            mask_length = mask.type_as(node_states).sum(dim=1).clamp_(1.0, 1e4)
            rv_node_out /= mask_length
        elif self.wnstack_norm == 'sqrt_mean':
            mask_length = mask.type_as(node_states).sum(dim=1).clamp_(1.0, 1e4).sqrt_()
            rv_node_out /= mask_length

        return rv_node_out

    def accumulate_rightward(self, node_states, mask, right_weight, tofloat=False, **kwargs):
        # node_states:      [b, n, m, c]
        # mask:             [b, n, m, 1]

        # node_states_t:    [b, c, m, n]
        # right_weight:     [b, 1, n, 1]
        # rv_node_out:      [b, m, c]
        if self.wnstack_norm == 'mean':
            mask_length = mask.type_as(node_states).sum(dim=1, keepdim=True).clamp_(1.0, 1e4)
            node_states /= mask_length
        elif self.wnstack_norm == 'sqrt_mean':
            mask_length = mask.type_as(node_states).sum(dim=1, keepdim=True).clamp_(1.0, 1e4).sqrt_()
            node_states /= mask_length

        node_states_t = node_states.transpose(1, 3)
        # assert not torch.isnan(right_weight).any(), f'right_weight::right_weight problem: {right_weight}'

        if tofloat:
            rv_node_out = torch.matmul(node_states_t.float(), right_weight.float()).type_as(node_states_t)
        else:
            rv_node_out = torch.matmul(node_states_t, right_weight)
        # rv_node_out = torch.matmul(node_states_t, right_weight).squeeze_(-1).transpoAse(1, 2)
        # rv_node_out = rv_node_out.clamp_(-1e4, 1e4)
        rv_node_out = rv_node_out.squeeze_(-1).transpose(1, 2)

        # if torch.isnan(rv_node_out).any():
        #     print(f'Nan Occur!!!')
        #     rv_node_out_float = torch.matmul(node_states_t.float(), right_weight.float()).squeeze_(-1).transpose(1, 2)
        #     where_nan = rv_node_out_float[torch.isnan(rv_node_out)]
        #     print(where_nan)
        #     raise AssertionError('Nan in rv_node_out')
        # assert not torch.isnan(rv_node_out).any(), f'rv_node_out::after matmul problem: {rv_node_out}, {node_states_t.max()}, {node_states_t.min()}'
        assert not torch.isnan(rv_node_out).any(), f'rv_node_out::after matmul problem: NaN [{tofloat}][type={rv_node_out.dtype}], consider export value_tofloat=1 '
        # if self.wnstack_norm == 'mean':
        #     mask_length = mask.type_as(node_states).sum(dim=1).clamp_(1.0, 1e4)
        #     rv_node_out /= mask_length
        # elif self.wnstack_norm == 'sqrt_mean':
        #     mask_length = mask.type_as(node_states).sum(dim=1).clamp_(1.0, 1e4).sqrt_()
        #     rv_node_out /= mask_length

        return rv_node_out

    def _compute_nstree_states(self, leaves, rv_nodes, right_weight, mask, hier_embed=None, **kwargs):


        """alright folks this is the fucking main thing, alright! HIERARCHICAL ACCUMULATION"""
        leaves = leaves.unsqueeze(2)
        rv_nodes = rv_nodes.unsqueeze(1)
        # leaves:           [bh, n, c]
        # rv_nodes:         [bh, m, c]
        # right_weight:     [bh, 1, n, 1]
        # mask:             [bh, n, m, 1]
        # hier_embed:       [bh, n, m, c]
        # leaves:           [bh, n, 1, c]
        # rv_nodes:         [bh, 1, m, c]

        nodes = rv_nodes
        """HIERARCHICAL EMBEDDINGS"""
        if hier_embed is not None:
            nodes = rv_nodes + hier_embed
        """UPWARD CA"""
        node_states = self.accumulate_upward(leaves, nodes, mask, **kwargs)
        # assert not torch.isnan(node_states).any(), f'node_states::upward problem: {node_states.sum(-1)}'
        """WEIGHTED AGGREGATION"""
        rv_node_out = self.accumulate_rightward(
            node_states, mask, right_weight, tofloat=value_tofloat or self.value_tofloat_mle)
        return rv_node_out

    def compute_att_weights_values(
            self, q, k, v, node_k, node_v, ntree_mask, right_weight, hier_embed=None, force_self_att=False, **kwargs):
        # q                 [bh, t, d]
        # k:                [bh, n, d]
        # v:                [bh, n, d]
        # node_k:           [bh, m, d]
        # node_v:           [bh, m, d]
        # ntree_mask:       [bh, n, m, 1]
        attn_le = torch.bmm(q, k.transpose(1, 2))
        attn_no = torch.bmm(q, node_k.transpose(1, 2))
        # attn_le:          [bh, t, n]
        # attn_no:          [bh, t, m]
        # attn_le_t:        [bh, n, t]
        # attn_no_t:        [bh, m, t]
        # nstree_attn_wgts: [bh, t, n+m]

        if force_self_att and skip_aff_ln and self.nstack_mask_fname == MergeWeightMask.LEAVES_SUBTREE:
            assert q.size(1) == k.size(1) + node_k.size(1), f'{q.size(1)} != {k.size(1) + node_k.size(1)}'
            n = k.size(1)
            no_attn_le = attn_le[:, n:]
            no_attn_no = attn_no[:, n:]
            le_attn_le = attn_le[:, :n]
            le_attn_no = attn_no[:, :n]
            # no_attn_le:   [bh, m, n]
            # no_attn_no:   [bh, m, m]
            # le_attn_le:   [bh, n, n]
            # le_attn_no:   [bh, m, m]

            # no_attn_le_t: [bh, n, m]
            # no_attn_no_t: [bh, m, m]

            no_attn_le_t = no_attn_le.transpose(1, 2)
            no_attn_no_t = no_attn_no.transpose(1, 2)
            no_nstree_attn_no = self._compute_nstree_states(no_attn_le_t, no_attn_no_t, right_weight, ntree_mask, None)
            no_nstree_attn_no = no_nstree_attn_no.transpose(1, 2)
            # no_nstree_attn_no:    [bh, m, m]
            no_nstree_attn_wgts = torch.cat((no_attn_le, no_nstree_attn_no), 2)
            le_attn_wgts = torch.cat((le_attn_le, le_attn_no), 2)
            nstree_attn_wgts = torch.cat((le_attn_wgts, no_nstree_attn_wgts), 1)
            # no_nstree_attn_wgts:  [bh, n + m, n + m]
        else:
            attn_le_t = attn_le.transpose(1, 2)
            attn_no_t = attn_no.transpose(1, 2)

            nstree_attn_no = self._compute_nstree_states(attn_le_t, attn_no_t, right_weight, ntree_mask, None)
            nstree_attn_wgts = torch.cat((attn_le_t, nstree_attn_no), 1).transpose(1, 2)

        # torch.set_printoptions(profile="full")
        # assert not torch.isnan(node_v).any(), f'node_v::before nstack'
        # assert not torch.isnan(v).any(), f'v::before nstack'
        node_v = self._compute_nstree_states(v, node_v, right_weight, ntree_mask, hier_embed)
        # FIXME: values explode !
        assert not torch.isnan(node_v).any(), f'node_v::after nstack: {node_v.sum(-1)}'
        # torch.set_printoptions(profile="default")

        values = torch.cat([v, node_v], 1)
        return nstree_attn_wgts, values

    def compute_nstack_att(
            self, q, k, v, node_k, node_v, ntree_mask, right_weight, hier_embed, pad_mask, need_weights, force_self_att=False, **kwargs):
        # q                 [bh, t, d]
        # k:                [bh, n, d]
        # v:                [bh, n, d]
        # node_k:           [bh, m, d]
        # node_v:           [bh, m, d]
        # ntree_mask:       [bh, n, m, 1]
        # pad_mask:         [b, 1, n + m, n + m]
        bh, t, d = q.size()
        bh_, n, _ = k.size()
        bh__, m, _ = node_k.size()
        b = bh // self.num_heads

        attn_weights, values = self.compute_att_weights_values(
            q, k, v, node_k, node_v, ntree_mask, right_weight, hier_embed, force_self_att=force_self_att, **kwargs
        )

        # assert not torch.isnan(attn_weights).any(), f'weights::after-beforemaksing'
        attn_weights = self.nstack_mask_func(self, attn_weights, pad_mask, **kwargs)
        # assert not torch.isnan(attn_weights).any(), f'weights::after-masking'
        attn_weights = utils.softmax(attn_weights, dim=-1).type_as(attn_weights)
        # assert not torch.isnan(attn_weights).any(), f'weights::after-softmax'
        if attn_weights.dtype == torch.float16:
            attn_weights[torch.isnan(attn_weights)] = 0.0
        else:
            attn_weights = torch.where(torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights)
        assert not torch.isnan(attn_weights).any(), f'weights::after-zeroing'
        assert not torch.isnan(values).any(), f'values::nan'

        attn_weights = self.dropout_layer(attn_weights)
        """this is the final equation 18"""
        attn = torch.bmm(attn_weights, values)
        attn = attn.transpose(0, 1).contiguous().view(t, b, self.embed_dim)
        assert not torch.isnan(attn).any(), f'before outprof'
        assert not torch.isinf(attn).any(), f'before outprof'
        attn = self.out_proj(attn)
        assert not torch.isnan(attn).any(), f'after outprof'
        assert not torch.isinf(attn).any(), f'after outprof'

        if need_weights:
            attn_weights = attn_weights.view(b, self.num_heads, t, n + m)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights

    def forward(
            self, query, key, value, node_key, node_value, ntree_mask, hier_embed=None, pad_mask=None,
            key_pad=None, node_pad=None, incremental_state=None,
            need_weights=True, static_kv=False, attn_mask=None, force_self_att=False
    ):
        """t=m+n"""
        # query:            [t, b, c]
        # key:              [n, b, c]
        # value:            [n, b, c]
        # node_key:         [m, b, c]
        # node_value:       [m, b, c]
        # ntree_mask:       [bh, n, m, 1]
        # hier_embed:       [bh, n, m, d] c=d*h
        # pad_mask:         [b, 1, n + m, n + m]

        t, b, c = query.size()
        n, bk, c_ = key.size()
        m, b__, c__ = node_key.size()

        h = self.num_heads
        if key_pad is None:
            assert node_pad is None

        assert attn_mask is None

        (q, k, v, node_k, node_v, key_padding_mask, node_padding_mask,
         saved_state, src_len, node_src_len, tgt_len, query_bsz) = self.prepare_dptree_qkv(
            query, key, value, node_key, node_value, key_pad, node_pad, incremental_state,
            need_weights, static_kv, True, True, force_self_att
        )

        right_weight = self.project_dwstack_key(key)
        # right_weight:     [n, b, h]
        right_weight = right_weight.contiguous().view(n, 1, bk * h, 1).transpose(0, 2)
        # right_weight:     [bh, 1, n, 1]

        (attn, attn_weights) = self.compute_nstack_att(
            q, k, v, node_k, node_v, ntree_mask, right_weight, hier_embed, pad_mask, need_weights,
            force_self_att=force_self_att,
        )

        return attn, attn_weights

"""this is the main attention mechanism class"""
class MergeStackNodesOnValueAttention(MergeStackNodesOnAffinityValueAttention):
    def compute_att_weights_values(self, q, k, v, node_k, node_v, ntree_mask, right_weight, hier_embed=None, **kwargs):
        # q                 [bh, t, d]
        # k:                [bh, n, d]
        # v:                [bh, n, d]
        # node_k:           [bh, m, d]
        # node_v:           [bh, m, d]
        # ntree_mask:       [bh, n, m, 1]
        attn_le = torch.bmm(q, k.transpose(1, 2))
        attn_no = torch.bmm(q, node_k.transpose(1, 2))
        # attn_le:          [bh, t, n]
        # attn_no:          [bh, t, m]
        # attn_le_t:        [bh, n, t]
        # attn_no_t:        [bh, m, t]
        # nstree_attn_wgts: [bh, t, n+m]
        # attn_le_t = attn_le.transpose(1, 2)
        # attn_no_t = attn_no.transpose(1, 2)

        # nstree_attn_no = self._compute_nstree_states(attn_le_t, attn_no_t, right_weight, ntree_mask, None)
        # nstree_attn_wgts = torch.cat([attn_le_t, nstree_attn_no], 1).transpose(1, 2)
        # nstree_attn_no = self._compute_nstree_states(attn_le_t, attn_no_t, right_weight, ntree_mask, None)
        nstree_attn_wgts = torch.cat([attn_le, attn_no], 2)

        # torch.set_printoptions(profile="full")
        # assert not torch.isnan(node_v).any(), f'node_v::before nstack'
        # assert not torch.isnan(v).any(), f'v::before nstack'
        node_v = self._compute_nstree_states(v, node_v, right_weight, ntree_mask, hier_embed)
        # FIXME: values explode !
        assert not torch.isnan(node_v).any(), f'node_v::after nstack: {node_v.sum(-1)}'
        # torch.set_printoptions(profile="default")
        values = torch.cat([v, node_v], 1)
        return nstree_attn_wgts, values
