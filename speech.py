import torch
from transformers import GPT2Model, GPT2Tokenizer

# Path to your pretrained model
model_path = "./models/shakespeare_model.pth"
# Load the model state_dict
model_state_dict = torch.load(model_path)
# Create an instance of GPT2Model
model = GPT2Model.from_pretrained(pretrained_model_name_or_path=None, state_dict=model_state_dict)

# My input
input_text = "Your input text goes here."
input_ids = tokenizer.encode(input_text, return_tensors="pt")

'''
with torch.no_grad():
    outputs = model(input_ids)
    embeddings = outputs.last_hidden_state
'''

output = model.generate(input_ids, max_length=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded_output)

