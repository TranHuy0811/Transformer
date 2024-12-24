import torch
import torch.nn as nn

class Positional_Encoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pos_encoding = torch.zeros((max_len, d_model))
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        _2i = torch.arange(0, d_model, step=2, dtype=torch.float).unsqueeze(0)
        angles = pos / 10000**(_2i / d_model)

        pos_encoding[:, 0::2] = torch.sin(angles)
        pos_encoding[:, 1::2] = torch.cos(angles)

        # (persistent = False) -> Không lưu giá trị này khi chạy state_dict
        self.register_buffer("pos_encoding", pos_encoding, persistent=False)


    def forward(self, seq_len): 
        return self.pos_encoding[:seq_len, :]