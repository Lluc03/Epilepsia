import torch
import torch.nn as nn
from models.models_init import init_weights_kaiming_normal  # ← CAMBIO CLAVE

class FeatureExtractor(nn.Module):
    """Extractor de características CNN1D"""
    def __init__(self, n_channels=21):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),  # T/4
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),  # T/16
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)   # T/32 → [B, 128, 4]
        )
        
        self.fusion = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        init_weights_kaiming_normal(self)  # ← Kaiming es mejor para ReLU
    
    def forward(self, x):
        x = self.conv(x)      # [B,21,128] → [B,128,4]
        x = self.fusion(x)    # → [B,64,1]
        x = torch.flatten(x, 1)  # → [B,64]
        return x

class Classifier(nn.Module):
    """Clasificador simple"""
    def __init__(self, input_dim=64, n_classes=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
        init_weights_kaiming_normal(self)
    
    def forward(self, x):
        return self.fc(x)

class CNN1DSeparated(nn.Module):
    """Pipeline completo: Extractor + Clasificador"""
    def __init__(self, n_channels=21, n_classes=2):
        super().__init__()
        self.feature_extractor = FeatureExtractor(n_channels)
        self.classifier = Classifier(input_dim=64, n_classes=n_classes)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits
    
    def extract_features(self, x):
        """Solo extracción de características"""
        return self.feature_extractor(x)