import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using the {device}")

# Load and preprocess data
data = ""
with open('./input.txt', 'r', encoding='utf-8') as f:
    data = f.read()

# Tokenization
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenized_data = tokenizer(data)

# Create dataset
class CustomDataset(Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

dataset = CustomDataset(tokenized_data)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
loss_function = torch.nn.CrossEntropyLoss()

# Number of epochs
num_epochs = 5000  # You can adjust this number: 5000

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs = batch[:, :-1]  # Take all elements except the last one as inputs
        labels = batch[:, 1:]    # Take all elements except the first one as labels
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs.logits.view(-1, outputs.logits.size(-1)), labels.contiguous().view(-1))
        loss.backward()
        optimizer.step()

# Save trained model
torch.save(model.state_dict(), "./models/trained_model.pth")
