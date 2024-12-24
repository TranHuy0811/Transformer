import torch
import torch.nn as nn
import torch.nn.functional as F

class Multi_Head_Attention(nn.Module):
    def __init__(self, d_model, num_heads, drop_prob):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scaling = d_model**(-0.5)

        if (self.head_dim * num_heads) != d_model:
            raise ValueError(
                f"d_model must be divisible by num_heads (got `d_model`: {d_model}"
                f" and `num_heads`: {num_heads})."
            )
        
        self.value_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.query_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=drop_prob)


    def proj_reshape(self, var, batch, token_len): # (batch, token len, d_model) -> (batch, num_heads, token_len, head_dim)
        var = var.reshape(batch, token_len, self.num_heads, self.head_dim).transpose(1, 2)
        return var
    
    def inv_proj_reshape(self, var, batch, token_len): # (batch, num_heads, token_len, head_dim) -> (batch, token len, d_model)
        var = var.transpose(1, 2).reshape(batch, token_len, self.d_model)
        return var
        

    def forward(self, init_v, init_k, init_q, mask=None):
        # Separate token_len to handle Cross-Attention (token_len1 = input token_length, token_len2 = target token_length)
        batch, token_len1, _ = init_v.size()
        _, token_len2, _ =  init_q.size()

        value = self.value_proj(init_v) 
        key = self.key_proj(init_k) 
        query = self.query_proj(init_q) * self.scaling

        value = self.proj_reshape(value, batch, token_len1)
        key = self.proj_reshape(key, batch, token_len1)
        query = self.proj_reshape(query, batch, token_len2)

        att_weight = query @ key.transpose(2, 3)

        if mask is not None:
            att_weight = att_weight.masked_fill(mask == 0, float('-inf'))
            
        att_weight = F.softmax(att_weight, dim=-1)
        att_weight = self.dropout(att_weight)

        att_out = att_weight @ value
        att_out = self.inv_proj_reshape(att_out, batch, token_len2)

        att_out = self.out_proj(att_out)
        return att_out
        



        
