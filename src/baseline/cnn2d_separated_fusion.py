import torch
import torch.nn as nn
from models.models_init import init_weights_kaiming_normal

class FeatureExtractorFusion2D(nn.Module):
    """
    Feature Extractor con fusión a nivel de características
    Según diapositivas: Fusion at Feature Level
    """
    def __init__(self, n_channels=21, fusion_method='weighted'):
        """
        Args:
            n_channels: Número de canales EEG (21)
            fusion_method: 'concat' | 'average' | 'weighted'
        """
        super().__init__()
        self.n_channels = n_channels
        self.fusion_method = fusion_method
        
        # CONVOLUTIONAL UNIT (Conv2D para preservar dimensión de canales)
        # Input: [B, 1, 21, 128]
        self.conv = nn.Sequential(
            # Bloque 1: kernel=(1, 3) solo en dimensión temporal
            nn.Conv2d(1, 16, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),  # T/4 → [B, 16, 21, 32]
            
            # Bloque 2
            nn.Conv2d(16, 32, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),  # T/16 → [B, 32, 21, 8]
            
            # Bloque 3
            nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))   # T/32 → [B, 64, 21, 4]
        )
        
        # FUSION UNIT (fusionar los 21 canales)
        if fusion_method == 'concat':
            # Opción 1: Concatenación
            self.fusion = None
            self.output_features = 64 * n_channels * 4  # 64 * 21 * 4 = 5,376
            
        elif fusion_method == 'average':
            # Opción 2: Average pooling sobre canales
            self.fusion = nn.AvgPool2d(kernel_size=(n_channels, 1))  # [B, 64, 1, 4]
            self.output_features = 64 * 4  # 256
            
        elif fusion_method == 'weighted':
            # Opción 3: Weighted average (trainable) - RECOMENDADO
            self.fusion = nn.Sequential(
                # Necesitamos permutar: [B, 64, 21, 4] → [B, 21, 64, 4]
                # para que Conv2d fusione sobre la dimensión de canales
                nn.Conv2d(n_channels, 1, kernel_size=(1, 1)),  # [B, 1, 64, 4]
                nn.ReLU()
            )
            self.output_features = 64 * 4  # 256
        
        init_weights_kaiming_normal(self)
    
    def forward(self, x):
        """
        Args:
            x: [B, 21, 128] - Input EEG windows
        Returns:
            features: [B, output_features]
        """
        # Añadir dimensión de canal: [B, 21, 128] → [B, 1, 21, 128]
        x = x.unsqueeze(1)
        
        # Convolutional Unit: [B, 1, 21, 128] → [B, 64, 21, 4]
        x = self.conv(x)
        
        # Fusion Unit
        if self.fusion_method == 'concat':
            # Opción 1: Flatten
            x = torch.flatten(x, 1)  # [B, 64*21*4]
            
        elif self.fusion_method == 'average':
            # Opción 2: Average pooling
            x = self.fusion(x)  # [B, 64, 1, 4]
            x = torch.flatten(x, 1)  # [B, 256]
            
        elif self.fusion_method == 'weighted':
            # Opción 3: Weighted average
            # Permutar para fusionar sobre canales: [B, 64, 21, 4] → [B, 21, 64, 4]
            x = x.permute(0, 2, 1, 3)
            x = self.fusion(x)  # [B, 1, 64, 4]
            x = torch.flatten(x, 1)  # [B, 256]
        
        return x


class Classifier(nn.Module):
    """Clasificador simple"""
    def __init__(self, input_dim, n_classes=2):
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


class CNN2DSeparated(nn.Module):
    """
    Pipeline completo con fusión correcta según diapositivas
    """
    def __init__(self, n_channels=21, n_classes=2, fusion_method='weighted'):
        super().__init__()
        self.feature_extractor = FeatureExtractorFusion2D(n_channels, fusion_method)
        self.classifier = Classifier(
            input_dim=self.feature_extractor.output_features,
            n_classes=n_classes
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits
    
    def extract_features(self, x):
        """Solo extracción de características"""
        return self.feature_extractor(x)