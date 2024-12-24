import torch
import torch.nn as nn

from embedding.transformer_embedding import Embedding
from encoder_layer import Encoder_Layer

class Encoder(nn.Module):
    def __init__(self, num_blocks, d_model, vocab_size, max_len, num_heads, drop_prob, expand_factor=4, padding_idx=0):
        super().__init__()

        self.embedding = Embedding(vocab_size, d_model, max_len, padding_idx)
        self.dropout = nn.Dropout(p=drop_prob)
        self.blocks = nn.ModuleList(
            [Encoder_Layer(d_model, num_heads, drop_prob, expand_factor) for i in range(num_blocks)]
        )


    def forward(self, x, mask=None): # x: (batch, token len) => (batch, token len, d_model)
        x = self.embedding(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, mask)

        return x