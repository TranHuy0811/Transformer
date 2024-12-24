import torch
import torch.nn as nn

from embedding.transformer_embedding import Embedding
from decoder_layer import Decoder_Layer

class Decoder(nn.Module):
    def __init__(self, num_blocks, d_model, vocab_size, max_len, num_heads, drop_prob, expand_factor=4, padding_idx=0):
        super().__init__()

        self.embedding = Embedding(vocab_size, d_model, max_len, padding_idx)
        self.dropout = nn.Dropout(p=drop_prob)
        self.blocks = nn.ModuleList(
            [Decoder_Layer(d_model, num_heads, drop_prob, expand_factor) for i in range(num_blocks)]
        )


    def forward(self, x, enc_out, x_mask=None, src_mask=None):
        x = self.embedding(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, enc_out, x_mask, src_mask)

        return x