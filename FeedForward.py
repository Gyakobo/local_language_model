from header import * 

class FeedFowared(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLu(),
        )

    def forward(self, x):
        return self.net(x)



