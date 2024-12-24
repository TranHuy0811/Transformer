import torch 
import torch.nn as nn
import torch.nn.functional as F

from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, num_blocks, num_heads, expand_factor=4, drop_prob=0.1, padding_idx=0):
        super().__init__()
        self.padding_idx = padding_idx
        
        self.encoder = Encoder(num_blocks, d_model, vocab_size, max_len, num_heads, drop_prob, expand_factor, padding_idx)
        self.decoder = Decoder(num_blocks, d_model, vocab_size, max_len, num_heads, drop_prob, expand_factor, padding_idx)
        self.linear = nn.Linear(d_model, vocab_size)
    
    def padding_mask(self, seq):
        mask = (seq != self.padding_idx).unsqueeze(1).unsqueeze(2)
        return mask

    def future_mask(self, seq):
        seq_len = seq.size(-1)
        mask = torch.tril(torch.ones(seq_len, seq_len)).type(torch.ByteTensor).to(seq.device)
        mask = mask.unsqueeze(0).unsqueeze(1)
        return mask


    def forward(self, src, trg):
        src_mask = self.padding_mask(src)
        trg_mask = self.padding_mask(trg) & self.future_mask(trg)

        enc_out = self.encoder(src, mask=src_mask)
        dec_out = self.decoder(trg, enc_out, trg_mask, src_mask)
        out = self.linear(dec_out) 
        return out