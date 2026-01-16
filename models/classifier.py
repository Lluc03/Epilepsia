import torch.nn as nn

class EEGClassifier(nn.Module):
    def __init__(self, input_dim=128, num_classes=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)
