# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F


class PointerNet(nn.Module):
    def __init__(self, query_vec_size, src_encoding_size, attention_type='affine'):
        super(PointerNet, self).__init__()

        assert attention_type in ('affine', 'dot_prod')
        if attention_type == 'affine':
            self.src_encoding_linear = nn.Linear(src_encoding_size, query_vec_size, bias=False)

        self.attention_type = attention_type

    def forward(self, src_encodings, src_token_mask, query_vec):
        """
        :param src_encodings: Variable(batch_size, src_sent_len, hidden_size * 2)
        :param src_token_mask: Variable(batch_size, src_sent_len)
        :param query_vec: Variable(tgt_action_num, batch_size, query_vec_size)
        :return: Variable(tgt_action_num, batch_size, src_sent_len)
        """
        if self.attention_type == 'affine':
            src_encodings = self.src_encoding_linear(src_encodings)
        src_encodings = src_encodings.unsqueeze(1)

        q = query_vec.unsqueeze(3)

        weights = torch.matmul(src_encodings, q).squeeze(3)

        weights = weights.permute(1,0,2)
        if src_token_mask is not None:
            
            src_token_mask = src_token_mask.unsqueeze(0).expand_as(weights)
            weights.data.masked_fill_(src_token_mask, -float('inf'))
        weights = weights.permute(1,0,2)
        ptr_weights = F.softmax(weights, dim=-1)
        return ptr_weights
