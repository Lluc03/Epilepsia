import torch
import torch.nn as nn
from models.models_init import init_weights_xavier_normal

class CNN1DBaseline(nn.Module):
    """
    CNN1D amb fusió de canals DESPRÉS de la convolució
    (com a la diapositiva).
    """
    def __init__(self, n_channels=21, n_classes=2):
        super().__init__()

        # Convolutional Unit
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(4),   # T/4

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(4),   # T/16

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(4)    # T/64
        )

        # Channel Fusion at feature level
        # (Data Fusion Unit de la diapositiva)
        self.fusion = nn.Sequential(
            nn.Conv1d(64, 1, kernel_size=1),  # Fusió a 1 canal
            nn.ReLU()
        )

        # Output fully connected layers (128 → 2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

        init_weights_xavier_normal(self)

    def forward(self, x):  # x: [B,21,128]
        x = self.conv(x)         # → [B,64,T/64] ≈ [B,64,2]
        x = self.fusion(x)       # → [B,1,2]
        x = self.classifier(x)   # → [B,2]
        return x
