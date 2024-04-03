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

# print(encode("a"))
# print(encode("hii there"))
# print(decode(encode("hii there")))

# let's now encode the entire text dataset and store it into a torch.Tensor
import torch
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000]) # the 1000 characters we looked at earlier will look like this to the GPT 

# Let's now split up the data into train and validation sets
# This will help us understand to what extent our model is overfitting
n = int(0.9 * len(data)) # the first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

