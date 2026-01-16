import torch
import torch.nn as nn
from models.models_init import init_weights_xavier_normal


class EpilepsyLSTM(nn.Module):
    """
    LSTM-based seizure detection model
    Procesa ventanas EEG de forma secuencial respetando la temporalidad
    
    Input: [batch, n_channels, sequence_length] -> [B, 21, 128]
    Output: [batch, n_classes] -> [B, 2]
    """
    def __init__(self, n_channels=21, n_classes=2, hidden_size=128, num_layers=2, dropout=0.4):
        super().__init__()
        
        self.n_channels = n_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM: procesa la secuencia temporal de cada canal
        # AJUSTE: Aumentar dropout de 0.3 a 0.4 para reducir overfitting
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Clasificador: toma el último estado oculto
        # AJUSTE: Aumentar dropout y añadir capa adicional
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),  # Aumentado de 0.3 a 0.5
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.4),  # Nueva capa con dropout
            nn.Linear(32, n_classes)
        )
        
        # Inicializar pesos
        init_weights_xavier_normal(self)
    
    def forward(self, x):
        """
        Args:
            x: [batch, n_channels, sequence_length] = [B, 21, 128]
        Returns:
            logits: [batch, n_classes] = [B, 2]
        """
        # Reorganizar: [B, 21, 128] -> [B, 128, 21]
        # LSTM espera [batch, sequence_length, features]
        x = x.permute(0, 2, 1)  # [B, 128, 21]
        
        # LSTM processing
        # out: [B, 128, hidden_size] (salida de todos los timesteps)
        # hn, cn: [num_layers, B, hidden_size] (estados finales)
        out, (hn, cn) = self.lstm(x)
        
        # Tomar el último timestep: [B, hidden_size]
        last_output = out[:, -1, :]
        
        # Clasificación
        logits = self.classifier(last_output)
        
        return logits
    
    def extract_features(self, x):
        """
        Extrae features sin clasificar
        Útil para análisis o fine-tuning
        """
        x = x.permute(0, 2, 1)
        out, (hn, cn) = self.lstm(x)
        return out[:, -1, :]