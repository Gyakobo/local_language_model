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

# let's now encode the entire text dataset and store it into a torch.Tensor
import torch
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
# print(data[:1000]) # the 1000 characters we looked at earlier will look like this to the GPT 

# Let's now split up the data into train and validation sets
# This will help us understand to what extent our model is overfitting
n = int(0.9 * len(data)) # the first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

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

torch.manual_seed(1337) # Random number coefficient
batch_size = 4 # How many independent sequences will we process in parallel?
block_size = 8 # Maximum context length for predictions

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
'''
print('inputs:')
print(xb.shape)
print(xb)
print('target:') 
print(yb.shape)
print(yb)
print('-----')
'''

from bigramlm import *
m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

print(decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
