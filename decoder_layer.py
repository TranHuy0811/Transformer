import torch
import torch.nn as nn

from attention import Multi_Head_Attention
from feed_forward_network import Feed_Forward_Network

class Decoder_Layer(nn.Module):
    def __init__(self, d_model, num_heads, drop_prob, expand_factor=4):
        super().__init__()

        self.future_masked_att = Multi_Head_Attention(d_model, num_heads, drop_prob)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.layernorm1 = nn.LayerNorm(d_model)

        self.cross_att = Multi_Head_Attention(d_model, num_heads, drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.layernorm2 = nn.LayerNorm(d_model)

        self.ffn = Feed_Forward_Network(d_model, expand_factor,  drop_prob)
        self.dropout3 = nn.Dropout(p=drop_prob)
        self.layernorm3 = nn.LayerNorm(d_model)


    def forward(self, x, enc_out, x_mask=None, src_mask=None):
        new_x1 = self.future_masked_att(x, x, x, mask=x_mask)
        new_x1 = self.dropout1(new_x1)
        x = self.layernorm1(x + new_x1)

        new_x2 = self.cross_att(enc_out, enc_out, x, mask=src_mask)
        new_x2 = self.dropout2(new_x2)
        x = self.layernorm2(x + new_x2)

        new_x3 = self.ffn(x)
        new_x3 = self.dropout3(new_x3)
        x = self.layernorm3(x + new_x3)
        return x