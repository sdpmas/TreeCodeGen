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
from .attention import * 
from .embeddings import *
from fairseq.modules.multihead_attention import MultiheadAttention
import tensorflow as tf



class DecoderLayer(nn.Module):

    def __init__(self, args, no_encoder_attn=False):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        """the self attention for decoder is regular Multihead attention because there are no tress in the decoder input."""
        """important"""
        #TODO: might want to add a self attention layer.
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = getattr(args, 'relu_dropout', 0)
        self.input_dropout = getattr(args, 'input_dropout', 0)

        self.relu_dropoute_layer = nn.Dropout(self.relu_dropout)
        self.plain_dropoute_layer = nn.Dropout(self.dropout)
        self.input_dropout_layer = nn.Dropout(self.input_dropout)

        self.normalize_before = args.decoder_normalize_before
        self.dptree_class = args.dptree_class
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)


        self.nstack_cross = getattr(args, 'nstack_cross', True)
        

        if no_encoder_attn:
            print("we get here ")
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            # args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
            cross_kwargs = {
                'nstack_mask_fn': getattr(args, 'cross_nstack_mask_fn', args.nstack_mask_fn)
            }
            if self.nstack_cross:
                print(f'Build Cross attention: {self.dptree_class}')
                self.encoder_attn = self.dptree_class(
                    args, self.embed_dim, args.decoder_attention_heads,
                    dropout=args.attention_dropout, **cross_kwargs
                )
            else:
                """not the new cross attention but still cross multihead attention with the encoder output, which is kinda cool"""
                self.encoder_attn = MultiheadAttention(
                    self.embed_dim, args.decoder_attention_heads,
                    dropout=args.attention_dropout,
                )
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = True

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

    def forward(
            self, x, enc_le, enc_no,ntree_mask, hier_embed, pad_mask, key_pad, node_pad,
            incremental_state,
            prev_self_attn_state=None,
            prev_attn_state=None,
            self_attn_mask=None,
            self_attn_padding_mask=None
            # x_le, x_no, ntree_mask, hier_embed, pad_mask, key_pad, node_pad
    ):
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        x = self.input_dropout_layer(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.plain_dropoute_layer(x)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        attn = None
        need_weights = (not self.training and self.need_attn)
        # print(f'Cross:need_weights: {need_weights}/ {self.training} , {self.need_attn}')
        # self.nstack_cross=False
        if self.encoder_attn is not None:
            
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            if self.nstack_cross:
                # ntree_mask =
                # hier_embed =
                # pad_mask =
                x, attn = self.encoder_attn(
                    query=x,
                    key=enc_le, value=enc_le,
                    node_key=enc_no, node_value=enc_no,
                    ntree_mask=ntree_mask, hier_embed=hier_embed, pad_mask=pad_mask,
                    key_pad=key_pad, node_pad=node_pad, incremental_state=incremental_state,
                    need_weights=need_weights, static_kv=True
                )
            else:
                
                x, attn = self.encoder_attn(
                    query=x,
                    key=enc_le,
                    value=enc_le,
                    incremental_state=incremental_state,
                    static_kv=True,
                    need_weights=need_weights,
                )
            # x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.plain_dropoute_layer(x)
            x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.relu_dropoute_layer(x)
        x = self.fc2(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.plain_dropoute_layer(x)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        if self.onnx_trace:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
            return x, attn, self_attn_state
        return x, attn


class Decoder(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary,embed_tokens, no_encoder_attn=False, left_pad=False, final_norm=True):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed
        self.embed_dropout_layer = nn.Dropout(self.dropout)

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        output_embed_dim = args.decoder_output_dim
        #TODO: no hard coding
        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        # TODO: set params
        assert args.encoder_embed_dim == args.decoder_embed_dim, f'encoder-decoder dim not work !='
        assert args.encoder_attention_heads == args.decoder_attention_heads, f'decoder_att_heads !='
        self.heads = args.encoder_attention_heads
        self.encoder_embed_dim = args.encoder_embed_dim
        self.head_dim = self.encoder_embed_dim // self.heads

        # self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None

        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, padding_idx,
            left_pad=left_pad,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.nstack_cross = getattr(args, 'nstack_cross', True)
        self.nstack_hier_embed = getattr(args, 'nstack_hier_embed', False)
        self.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
        self.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
        self.nstack_hier_embed_share = getattr(args, 'nstack_hier_embed_share', False)
        self.hier_pos_positions = MergeHierarchicalEmbedding(
            args, args.encoder_layers, self.head_dim, self.heads,
            self.nstack_hier_embed_max_horiz, self.nstack_hier_embed_max_ver,
            self.nstack_hier_embed_share
        ) if self.nstack_hier_embed and self.nstack_cross else None

        self.nstack_mask_fname = getattr(args, 'cross_nstack_mask_fn', args.nstack_mask_fn)
        self.mutual_level = getattr(args, 'mutual_ancestor_level', 5)
        self.nstack_mask_building_func = MergeWeightMask.acquire_mask_building_fn(
            self.nstack_mask_fname, self.mutual_level)

        self.dptree_class = args.dptree_class

        self.layers = nn.ModuleList([])
        self.layers.extend([
            DecoderLayer(args, no_encoder_attn)
            for _ in range(args.decoder_layers)
        ])

        self.adaptive_softmax = None

        self.project_out_dim = Linear(embed_dim, output_embed_dim, bias=False) \
            if embed_dim != output_embed_dim and not args.tie_adaptive_weights else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), output_embed_dim))
            """it was on"""
            # nn.init.normal_(self.embed_out, mean=0, std=output_embed_dim ** -0.5)
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.decoder_normalize_before and final_norm
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)
    def get_timing_signal_1d(self, length,
                             channels,
                             min_timescale=1.0,
                             max_timescale=1.0e4,
                             start_index=0):
        position = tf.cast(tf.range(length) + start_index,dtype=float)
        num_timescales = channels // 2
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            tf.maximum(tf.cast(num_timescales,dtype=float) - 1, 1))
        inv_timescales = min_timescale * tf.exp(
            tf.cast(tf.range(num_timescales),dtype=float) * -log_timescale_increment)
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        signal = tf.pad(signal, [[0, 0], [0, tf.math.floormod(channels, 2)]])
        signal = tf.reshape(signal, [1, length, channels])
        return signal

    def forward(self, out_embed, encoder_out=None, incremental_state=None, **kwargs):
        
        positions=None
        
        """set embed scale to 1 for now"""
        out_embed= self.embed_scale * out_embed
        length=out_embed.shape[1]
        channels=out_embed.shape[2]
        pos = torch.from_numpy(self.get_timing_signal_1d(length,channels).numpy()).cuda()
        out_embed+=pos
        x=out_embed

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.embed_dropout_layer(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # todo: retrieving cross attention
        # 'encoder_out': x,  # T x B x C
        # 'encoder_indices': src_node_indices,  # B x T x 2
        # 'encoder_padding_mask': key_padding_mask,  # B x T
        # 'node_padding_mask': node_padding_mask,  # B x T
        # out_dict = {
        #     'encoder_out': x,  # (n + m) x b x C
        #     'encoder_indices': src_node_indices,  # B x m x 2
        #     'encoder_padding_mask': key_pad,  # B x n
        #     'node_padding_mask': node_pad,  # B x m
        # }
        """
        :param key:                 [tk, b, m, c]
        :param value:               [tk, b, m, c]
        :param node_key:            [nk, b, m, c]
        :param node_value:          [nk, b, m, c]
        :param indices:             [nk, b, m, 2]
        """

        assert encoder_out is not None, f'encoder_out is None!'
        encoder_output = encoder_out['encoder_out']
        spans = encoder_out['encoder_indices']
        encoder_pad = encoder_out['encoder_padding_mask']
        node_pad = encoder_out['node_padding_mask']

        nm, b_, c = encoder_output.size()
        b, m, _ = spans.size()
        # tk = tnk - nk
        h = self.heads
        n = nm - m
        encoder_leaves = encoder_output[:n]
        encoder_nodes = encoder_output[n:]

        inner_atts = []

        # ntree_mask =
        # hier_embed =
        # pad_mask =
        device = x.device
        ntree_mask = MergeStackNodesOnAffinityValueAttention.get_ntree_mask(n, spans, self.heads)
        pad_mask = self.nstack_mask_building_func(device, h, encoder_pad, node_pad, spans, b, x.size(0), n, m, **kwargs)
        hier_embeds = self.hier_pos_positions(n, spans) if self.nstack_hier_embed and self.nstack_cross else [None] * len(self.layers)
        # hier_embeds =[None] * len(self.layers)
        
        for i, (layer, hier_embed) in enumerate(zip(self.layers, hier_embeds)):
            # hier_embed = hier_embeds[i]
            x, attn = layer(
                x=x, enc_le=encoder_leaves, enc_no=encoder_nodes,
                ntree_mask=ntree_mask, hier_embed=hier_embed, pad_mask=pad_mask, key_pad=encoder_pad, node_pad=node_pad,
                incremental_state=incremental_state,
                prev_self_attn_state=None,
                prev_attn_state=None,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
            )
            inner_states.append(x)
            inner_atts.append(attn)

        if self.normalize:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        
        # print("the shape of the decoder out is ",x.shape)
        return x, {'attn': attn, 'inner_states': inner_states, 'inner_atts': inner_atts}

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    #TODO: make sure this thing works
    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = '{}.layers.{}.layer_norms.{}.{}'.format(name, i, old, m)
                    if k in state_dict:
                        state_dict['{}.layers.{}.{}.{}'.format(name, i, new, m)] = state_dict[k]
                        del state_dict[k]
        if utils.item(state_dict.get('{}.version'.format(name), torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict['{}.version'.format(name)] = torch.Tensor([1])

        return state_dict

