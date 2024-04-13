import torch
import torch.nn as nn
from torch.nn import functional as F


# Hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000 
eval_interval = 500
learning_rate = 3e-4
# eval_iters = 200
eval_iters = 10 
# ------

n_embd = 384
block_size = 8
n_head = 6
n_layer = 6
dropout = 0.2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Computing on: {device}")
