import torch
from torch.utils.data import DataLoader

from src.dataloader.eeg_dataset import EEGWindowDataset
from src.utils.metrics import evaluate


def train_baseline(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=30,
    lr=0.0005,
    batch_size=256,
    class_weights=None,
    early_stopping_patience=5,
    weight_decay=1e-4
):
    """
    Entrenamiento con class weights, early stopping y regularización L2
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"🖥️  Dispositiu d'entrenament: {device}")
    
    # Datasets
    train_ds = EEGWindowDataset(X_train, y_train)
    val_ds = EEGWindowDataset(X_val, y_val)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Optimizer con regularización L2 (weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Loss con pesos de clase
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        print(f"⚖️  Class weights: {class_weights.cpu().numpy()}")
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    print(f"🔧 Optimizer: Adam (lr={lr}, weight_decay={weight_decay})")
    
    # Early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    training_losses = []
    validation_losses = []
    training_accuracies = []
    
    # Entrenamiento
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            
            # Gradient clipping para estabilidad
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        training_losses.append(avg_train_loss)
        
        # Validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)
                logits = model(X)
                loss = criterion(logits, y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)
        
        val_metrics = evaluate(model, val_loader)
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["f1"]
        training_accuracies.append(val_acc)
        
        print(f"Època {epoch:02d} | TrainLoss={avg_train_loss:.4f} | ValLoss={avg_val_loss:.4f} | ValAcc={val_acc:.4f} | ValF1={val_f1:.4f}")
        
        # Early stopping basado en validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠️  Early stopping triggered at epoch {epoch}")
                break
    
    # Restaurar mejor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✅ Restored best model (val_loss={best_val_loss:.4f})")
    
    return training_losses, validation_losses, training_accuracies