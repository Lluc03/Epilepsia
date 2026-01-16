import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_curve, auc,
    confusion_matrix, precision_recall_curve
)
import torch

def evaluate(model, dataloader):
    model.eval()

    device = next(model.parameters()).device

    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "f1": f1,
        "auc": auc_score,
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm
    }


def find_optimal_threshold(model, dataloader, criterion='f1', min_sensitivity=0.65):
    """
    Encuentra el threshold óptimo para maximizar una métrica.
    
    Args:
        model: Modelo entrenado
        dataloader: DataLoader de validación/test
        criterion: 'f1' | 'sensitivity' | 'balanced' | 'f1_min_sens' | 'f1_balanced'
        min_sensitivity: Sensibilidad mínima requerida (default 0.65)
    
    Returns:
        best_threshold: Threshold óptimo
        best_metrics: Métricas con ese threshold
    """
    model.eval()
    device = next(model.parameters()).device
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            
            logits = model(X)
            probs = torch.softmax(logits, dim=1)[:, 1]  # Probabilidad clase 1 (seizure)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    
    # Verificar que hay ejemplos de ambas clases
    n_positive = np.sum(y_true == 1)
    n_negative = np.sum(y_true == 0)
    
    if n_positive == 0 or n_negative == 0:
        print(f"\n⚠️  WARNING: Dataset desbalanceado extremo!")
        print(f"   Positivos: {n_positive}, Negativos: {n_negative}")
        print(f"   Usando threshold por defecto: 0.5")
        return 0.5, None
    
    # AJUSTE: Probar más thresholds y en rango más amplio
    thresholds = np.linspace(0.1, 0.9, 400)  # Aumentado de 200 a 400
    best_threshold = 0.5
    best_score = -1
    best_metrics = None
    
    print(f"\n{'='*70}")
    print(f"🔍 BUSCANDO THRESHOLD ÓPTIMO (criterio: {criterion})")
    print(f"{'='*70}")
    print(f"  Dataset: {len(y_true)} muestras")
    print(f"  Positivos (seizure): {n_positive} ({n_positive/len(y_true)*100:.1f}%)")
    print(f"  Negativos (normal):  {n_negative} ({n_negative/len(y_true)*100:.1f}%)")
    print(f"  Min sensitivity required: {min_sensitivity:.2f}")
    
    valid_thresholds_found = 0
    candidates = []  # Guardar los mejores candidatos
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        
        # Verificar que la predicción no es trivial
        n_pred_positive = np.sum(y_pred == 1)
        if n_pred_positive == 0 or n_pred_positive == len(y_pred):
            continue
        
        valid_thresholds_found += 1
        
        # Calcular métricas
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        # Calcular sensibilidad y especificidad
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Criterio de selección
        if criterion == 'f1':
            score = f1
        elif criterion == 'sensitivity':
            score = sensitivity
        elif criterion == 'balanced':
            # Balance entre sensibilidad y especificidad
            score = (sensitivity + specificity) / 2
        elif criterion == 'f1_min_sens':
            # F1 pero con sensibilidad mínima garantizada
            if sensitivity >= min_sensitivity:
                score = f1
                candidates.append({
                    'threshold': thresh,
                    'f1': f1,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'precision': precision,
                    'score': score
                })
            else:
                continue
        elif criterion == 'f1_balanced':
            # NUEVO: Balancear F1 con especificidad para reducir FP
            if sensitivity >= min_sensitivity:
                # Penalizar baja especificidad
                score = f1 * (0.7 + 0.3 * specificity)
                candidates.append({
                    'threshold': thresh,
                    'f1': f1,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'precision': precision,
                    'score': score
                })
            else:
                continue
        else:
            score = f1
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
            best_metrics = {
                'threshold': thresh,
                'accuracy': acc,
                'f1': f1,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'precision': precision,
                'confusion_matrix': cm
            }
    
    # Verificar si encontramos algún threshold válido
    if best_metrics is None:
        print(f"\n⚠️  WARNING: No se encontró threshold con criterio '{criterion}' (min_sens={min_sensitivity})")
        print(f"   Thresholds válidos probados: {valid_thresholds_found}/{len(thresholds)}")
        
        # AJUSTE: Si no encuentra con min_sensitivity, buscar el mejor sin restricción
        if len(candidates) == 0 and 'min_sens' in criterion:
            print(f"   Buscando mejor threshold SIN restricción de min_sensitivity...")
            
            for thresh in thresholds:
                y_pred = (y_prob >= thresh).astype(int)
                n_pred_positive = np.sum(y_pred == 1)
                if n_pred_positive == 0 or n_pred_positive == len(y_pred):
                    continue
                
                f1 = f1_score(y_true, y_pred, zero_division=0)
                cm = confusion_matrix(y_true, y_pred)
                tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                # Buscar el que maximiza F1 con al menos 50% sensitivity
                if sensitivity >= 0.5:
                    score = f1 * (0.7 + 0.3 * specificity)
                    if score > best_score:
                        best_score = score
                        best_threshold = thresh
                        best_metrics = {
                            'threshold': thresh,
                            'accuracy': accuracy_score(y_true, y_pred),
                            'f1': f1,
                            'sensitivity': sensitivity,
                            'specificity': specificity,
                            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                            'confusion_matrix': cm
                        }
        
        # Si aún no encuentra, usar 0.5
        if best_metrics is None:
            print(f"   Usando threshold por defecto: 0.5")
            y_pred = (y_prob >= 0.5).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            best_threshold = 0.5
            best_metrics = {
                'threshold': 0.5,
                'accuracy': accuracy_score(y_true, y_pred),
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'confusion_matrix': cm
            }
    
    print(f"\n✅ THRESHOLD ÓPTIMO ENCONTRADO:")
    print(f"  Threshold:      {best_threshold:.3f}")
    print(f"  Accuracy:       {best_metrics['accuracy']:.4f}")
    print(f"  F1-Score:       {best_metrics['f1']:.4f}")
    print(f"  Precision:      {best_metrics['precision']:.4f}")
    print(f"  Sensibilidad:   {best_metrics['sensitivity']:.4f}")
    print(f"  Especificidad:  {best_metrics['specificity']:.4f}")
    print(f"\n  Matriz de confusión:")
    print(f"    {best_metrics['confusion_matrix']}")
    
    # Mostrar top 3 candidatos si los hay
    if len(candidates) > 1:
        print(f"\n  📊 Top 3 candidatos:")
        sorted_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)[:3]
        for i, cand in enumerate(sorted_candidates, 1):
            print(f"     {i}. Thresh={cand['threshold']:.3f} | F1={cand['f1']:.3f} | "
                  f"Sens={cand['sensitivity']:.3f} | Spec={cand['specificity']:.3f}")
    
    print(f"{'='*70}\n")
    
    return best_threshold, best_metrics
    """
    Encuentra el threshold óptimo para maximizar una métrica.
    
    Args:
        model: Modelo entrenado
        dataloader: DataLoader de validación/test
        criterion: 'f1' | 'sensitivity' | 'balanced' | 'f1_min_sens'
        min_sensitivity: Sensibilidad mínima requerida (default 0.7)
    
    Returns:
        best_threshold: Threshold óptimo
        best_metrics: Métricas con ese threshold
    """
    model.eval()
    device = next(model.parameters()).device
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            
            logits = model(X)
            probs = torch.softmax(logits, dim=1)[:, 1]  # Probabilidad clase 1 (seizure)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    
    # Verificar que hay ejemplos de ambas clases
    n_positive = np.sum(y_true == 1)
    n_negative = np.sum(y_true == 0)
    
    if n_positive == 0 or n_negative == 0:
        print(f"\n⚠️  WARNING: Dataset desbalanceado extremo!")
        print(f"   Positivos: {n_positive}, Negativos: {n_negative}")
        print(f"   Usando threshold por defecto: 0.5")
        return 0.5, None
    
    # Probar diferentes thresholds
    thresholds = np.linspace(0.05, 0.95, 200)
    best_threshold = 0.5
    best_score = -1
    best_metrics = None
    
    print(f"\n{'='*70}")
    print(f"🔍 BUSCANDO THRESHOLD ÓPTIMO (criterio: {criterion})")
    print(f"{'='*70}")
    print(f"  Dataset: {len(y_true)} muestras")
    print(f"  Positivos (seizure): {n_positive} ({n_positive/len(y_true)*100:.1f}%)")
    print(f"  Negativos (normal):  {n_negative} ({n_negative/len(y_true)*100:.1f}%)")
    
    valid_thresholds_found = 0
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        
        # Verificar que la predicción no es trivial
        n_pred_positive = np.sum(y_pred == 1)
        if n_pred_positive == 0 or n_pred_positive == len(y_pred):
            continue
        
        valid_thresholds_found += 1
        
        # Calcular métricas
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        # Calcular sensibilidad y especificidad
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Criterio de selección
        if criterion == 'f1':
            score = f1
        elif criterion == 'sensitivity':
            score = sensitivity
        elif criterion == 'balanced':
            # Balance entre sensibilidad y especificidad
            score = (sensitivity + specificity) / 2
        elif criterion == 'f1_min_sens':
            # F1 pero con sensibilidad mínima garantizada
            if sensitivity >= min_sensitivity:
                score = f1
            else:
                continue
        else:
            score = f1
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
            best_metrics = {
                'threshold': thresh,
                'accuracy': acc,
                'f1': f1,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'confusion_matrix': cm
            }
    
    # Verificar si encontramos algún threshold válido
    if best_metrics is None:
        print(f"\n⚠️  WARNING: No se encontró threshold que cumpla criterio '{criterion}'")
        print(f"   Thresholds válidos probados: {valid_thresholds_found}/{len(thresholds)}")
        print(f"   Usando threshold por defecto: 0.5")
        
        # Calcular métricas con threshold 0.5
        y_pred = (y_prob >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        best_threshold = 0.5
        best_metrics = {
            'threshold': 0.5,
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'confusion_matrix': cm
        }
    
    print(f"\n✅ THRESHOLD ÓPTIMO ENCONTRADO:")
    print(f"  Threshold:      {best_threshold:.3f}")
    print(f"  Accuracy:       {best_metrics['accuracy']:.4f}")
    print(f"  F1-Score:       {best_metrics['f1']:.4f}")
    print(f"  Sensibilidad:   {best_metrics['sensitivity']:.4f}")
    print(f"  Especificidad:  {best_metrics['specificity']:.4f}")
    print(f"\n  Matriz de confusión:")
    print(f"    {best_metrics['confusion_matrix']}")
    print(f"{'='*70}\n")
    
    return best_threshold, best_metrics


def evaluate_with_threshold(model, dataloader, threshold=0.5):
    """
    Evalúa el modelo con un threshold específico.
    """
    model.eval()
    device = next(model.parameters()).device
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            
            logits = model(X)
            probs = torch.softmax(logits, dim=1)[:, 1]
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    y_pred = (y_prob >= threshold).astype(int)
    
    # Métricas
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = auc(fpr, tpr)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    
    # Sensibilidad, especificidad y precisión
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # AÑADIDO
    
    return {
        "threshold": threshold,
        "accuracy": acc,
        "f1": f1,
        "auc": auc_score,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,  # AÑADIDO
        "fpr": fpr,
        "tpr": tpr,
        "precision_curve": precision_curve,  # Renombrado para evitar conflicto
        "recall": recall_curve,
        "confusion_matrix": cm
    }