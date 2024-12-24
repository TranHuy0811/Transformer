import torch
import torch.nn as nn
import torch.nn.functional as F

class Feed_Forward_Network(nn.Module):
    def __init__(self, d_model, expand_factor, drop_prob):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_model * expand_factor)
        self.dropout = nn.Dropout(p=drop_prob)
        self.linear2 = nn.Linear(d_model * expand_factor, d_model)


    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x