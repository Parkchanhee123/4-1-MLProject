import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=16000, output_dim=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.fc(x)
