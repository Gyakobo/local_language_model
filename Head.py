import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

n_embd = 32
block_size = 8
device = 'cude' if torch.cuda.is_available() else 'cpu'

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # Compute attention score ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('inf'))
        wei = F.softmax(wei, dim=-1)

        # Perform the weightened aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out
