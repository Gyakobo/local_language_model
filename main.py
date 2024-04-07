# All the imports and global variables
from bigramlm import *

# Hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3
eval_iters = 200
# ------

torch.manual_seed(1337) # Random number coefficient

# Read it in to inspect it
with open('input.txt', 'r', encoding ='utf-8') as f:
    text = f.read()

# Here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# Create a mapping from characters to integers for both encoding and decoding
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: takes a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: takes a list of integers, output a string

# print(encode("hii there"))
# print(decode(encode("hii there")))

# Trian and test splits
# let's now encode the entire text dataset and store it into a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
# print(data[:1000]) # the 1000 characters we looked at earlier will look like this to the GPT 

# Let's now split up the data into train and validation sets
# This will help us understand to what extent our model is overfitting
n = int(0.9 * len(data)) # the first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# Simple example of dividing data into splits
'''
#tain_data[:block_size+1] # => ([ 18, 47, 56, 57, 1, 15, 47, 58 ])

# Shows a truncating example
block_size = 8
x = train_data[:block_size] # Input to the transformer
y = train_data[1:block_size+1] # The next block size characters
for t in range(block_size):
    content = x[:t+1]
    target = y[t]
    print(f"when input is {content} the target: {target}")
'''

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# xb, yb = get_batch('train')
'''
print('inputs:')
print(xb.shape)
print(xb)
print('target:') 
print(yb.shape)
print(yb)
print('-----')
'''

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"set {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))


