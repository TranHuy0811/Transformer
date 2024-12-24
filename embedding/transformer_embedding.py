import torch
import torch.nn as nn

from .positional_encoding import Positional_Encoding

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_encoding = Positional_Encoding(d_model, max_len)


    def forward(self, x): # Shape (batch, token len) -> (batch, token len, d_model)
        x = self.embedding(x) + self.pos_encoding(x.size(-1))
        return x