import torch
import pickle 
from transformers import GPT2Model, GPT2Tokenizer, GPT2LMHeadModel
from main import *

# Path to your pretrained model
model_path = "./models/shakespeare_model.pth"

# 1: Save the BigramLanguageModel
# with open(model_path, 'rb') as f:
#    bigram_model = pickle.load(f)

# Load the model state_dict
# model_state_dict = torch.load(model_path)
# Create an instance of GPT2Model
# model = GPT2Model.from_pretrained(pretrained_model_name_or_path=None, state_dict=model_state_dict)

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

flag = True
# My input
while flag:
    input_text = input("input: ") 
    if input_text == "exit":
        flag = False
        continue

    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    output = model.generate(input_ids, max_length=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True).split('.')[0]
    print(decoded_output, end="\n\n")

