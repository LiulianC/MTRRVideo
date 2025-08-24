"""Mock mamba_ssm for testing purposes"""
import torch
import torch.nn as nn

class Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        
        # Simple mock implementation using a linear layer
        self.linear = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # x: (B, L, D) or (B, D, L)
        if x.dim() == 3:
            return self.linear(x)
        else:
            return self.linear(x.transpose(-1, -2)).transpose(-1, -2)