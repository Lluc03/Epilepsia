import yaml
import os
import datetime
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.baseline.cnn2d_separated_fusion import CNN2DSeparated
from src.baseline.train_baseline import train_baseline
from src.dataloader.eeg_dataset import EEGWindowDataset
from src.utils.data_utils import (
    load_patients_metadata,
    load_patient_data,
    balance_train_data,
    temporal_train_val_split
)
from src.utils.metrics import evaluate, find_optimal_threshold, evaluate_with_threshold
from src.utils.visualizer import (
    save_roc, save_confusion_matrix,
    save_precision_recall, save_training_curves
)

if __name__ == "__main__":
    
    # CONFIGURACIÓN
    with open("src/config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    DATASET_PATH = config["dataset_path"]
    
    # Crear directorio de resultados
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    results_dir = f"results/kfold_lopo_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("🧠 CLASSIFICADOR D'EPILÈPSIA – K-FOLD LOPO")
    print("   Pipeline: Extractor de característiques → Classificador")
    print("   Estratègia: 5-Fold Leave-One-Patient-Out")
    print("=" * 70)
    
    # 1. CARGAR METADATOS
    patients_df = load_patients_metadata(DATASET_PATH)
    
    # 2. SELECCIONAR 5 PACIENTES PEDIÁTRICOS PARA TEST
    # Criterio: Diversidad en cantidad de crisis
    pediatric = patients_df[patients_df['is_pediatric'] == True].copy()
    
    # Seleccionar 5 pacientes representativos:
    # - 1 con muchas crisis
    # - 2 con crisis medias
    # - 2 con pocas crisis
    pediatric_sorted = pediatric.sort_values('n_seizure', ascending=False)
    
    test_patients_ids = [
        pediatric_sorted.iloc[0]['patient_id'],   # Más crisis (chb15 o similar)
        pediatric_sorted.iloc[5]['patient_id'],   # Crisis medias-altas
        pediatric_sorted.iloc[10]['patient_id'],  # Crisis medias
        pediatric_sorted.iloc[15]['patient_id'],  # Crisis medias-bajas
        pediatric_sorted.iloc[-5]['patient_id']   # Pocas crisis
    ]
    
    print(f"\n📋 PACIENTES SELECCIONADOS PARA K-FOLD:")
    for i, pid in enumerate(test_patients_ids, 1):
        pat_info = patients_df[patients_df['patient_id'] == pid].iloc[0]
        print(f"  Fold {i}: {pid} - {pat_info['n_seizure']:,} crisis ({pat_info['seizure_minutes']:.1f} min)")
    
    # Almacenar resultados de cada fold
    all_results = []
    
    # ==========================================
    # K-FOLD LOOP
    # ==========================================
    for fold, test_patient_id in enumerate(test_patients_ids, 1):
        print(f"\n{'='*70}")
        print(f"🔄 FOLD {fold}/{len(test_patients_ids)} - TEST PATIENT: {test_patient_id}")
        print(f"{'='*70}")
        
        # Split train/test
        test_patient = patients_df[patients_df['patient_id'] == test_patient_id]
        train_patients = patients_df[patients_df['patient_id'] != test_patient_id].copy()
        
        print(f"\n  TRAIN: {len(train_patients)} pacientes")
        print(f"  TEST:  {test_patient_id}")
        
        # 3. CARGAR DATOS TRAIN
        X_train_list, y_train_list = [], []
        for _, row in train_patients.iterrows():
            X, y = load_patient_data(row['npz_path'], row['meta_path'])
            X_train_list.append(X)
            y_train_list.append(y)
        
        X_train = np.concatenate(X_train_list)
        y_train = np.concatenate(y_train_list)
        
        # 4. BALANCEAR TRAIN
        X_train, y_train = balance_train_data(
            X_train, y_train,
            strategy='mixed',
            target_hours=40
        )
        
        # 5. SPLIT TRAIN/VAL
        X_train_final, X_val, y_train_final, y_val = temporal_train_val_split(
            X_train, y_train, val_ratio=0.15
        )
        
        # 6. CARGAR TEST
        X_test, y_test = load_patient_data(
            test_patient['npz_path'].values[0],
            test_patient['meta_path'].values[0]
        )
        
        print(f"\n  📦 DATOS:")
        print(f"     TRAIN: {X_train_final.shape}")
        print(f"     VAL:   {X_val.shape}")
        print(f"     TEST:  {X_test.shape}")
        
        # 7. MODELO
        model = CNN2DSeparated(
            n_channels=21,
            n_classes=2,
            fusion_method='weighted'  # ← o 'average' / 'concat'
        )
        
        # 8. CALCULAR CLASS WEIGHTS (CRÍTICO para recall)
        # Penalizar MÁS los falsos negativos (clase 1 = seizure)
        n_normal = np.sum(y_train_final == 0)
        n_seizure = np.sum(y_train_final == 1)
        
        # Peso inversamente proporcional a la frecuencia
        # Pero multiplicamos x3 el peso de seizure para forzar detección
        weight_normal = 1.0
        weight_seizure = (n_normal / n_seizure) * 3.0  # ← FACTOR 3 para priorizar recall
        
        class_weights = [weight_normal, weight_seizure]
        
        print(f"\n  ⚖️  Class weights calculados:")
        print(f"     Normal (0): {weight_normal:.2f}")
        print(f"     Seizure (1): {weight_seizure:.2f}")
        
        # 9. ENTRENAMIENTO
        training_losses, val_losses, training_accs = train_baseline(
            model,
            X_train_final,
            y_train_final,
            X_val,
            y_val,
            epochs=30,
            lr=0.0005,
            batch_size=256,
            class_weights=class_weights,  # ← NUEVO
            early_stopping_patience=5     # ← NUEVO
        )
        
        # 10. EVALUACIÓN EN TEST
        test_ds = EEGWindowDataset(X_test, y_test)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
        
        # Evaluar con threshold por defecto
        metrics_default = evaluate(model, test_loader)
        
        # Encontrar threshold óptimo en VALIDACIÓN
        val_ds = EEGWindowDataset(X_val, y_val)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
        
        optimal_threshold, _ = find_optimal_threshold(
            model, val_loader,
            criterion='f1_min_sens',
            min_sensitivity=0.70
        )
        
        # Evaluar en TEST con threshold óptimo
        metrics_optimized = evaluate_with_threshold(model, test_loader, optimal_threshold)
        
        # Guardar resultados del fold
        fold_results = {
            'fold': fold,
            'test_patient': test_patient_id,
            'n_test_samples': len(X_test),
            'n_test_seizures': np.sum(y_test == 1),
            'optimal_threshold': optimal_threshold,
            'accuracy_default': metrics_default['accuracy'],
            'f1_default': metrics_default['f1'],
            'sensitivity_default': metrics_default['confusion_matrix'][1, 1] / np.sum(y_test == 1),
            'accuracy_opt': metrics_optimized['accuracy'],
            'f1_opt': metrics_optimized['f1'],
            'sensitivity_opt': metrics_optimized['sensitivity'],
            'specificity_opt': metrics_optimized['specificity'],
            'auc': metrics_optimized['auc'],
            'confusion_matrix': metrics_optimized['confusion_matrix']
        }
        
        all_results.append(fold_results)
        
        print(f"\n✅ RESULTATS FOLD {fold}:")
        print(f"  Pacient: {test_patient_id}")
        print(f"  Threshold: {optimal_threshold:.3f}")
        print(f"  Accuracy:      {metrics_optimized['accuracy']:.4f}")
        print(f"  F1-Score:      {metrics_optimized['f1']:.4f}")
        print(f"  Sensibilitat:  {metrics_optimized['sensitivity']:.4f} ⭐")
        print(f"  Especificitat: {metrics_optimized['specificity']:.4f}")
        print(f"\n  Matriu de confusió:")
        print(f"    {metrics_optimized['confusion_matrix']}")
        
        # Guardar visualizaciones del fold
        fold_dir = os.path.join(results_dir, f"fold_{fold}_{test_patient_id}")
        os.makedirs(fold_dir, exist_ok=True)
        
        save_roc(metrics_optimized["fpr"], metrics_optimized["tpr"], 
                 metrics_optimized["auc"], fold_dir)
        save_confusion_matrix(metrics_optimized["confusion_matrix"], fold_dir)
        save_precision_recall(metrics_optimized["precision"], 
                             metrics_optimized["recall"], fold_dir)
        save_training_curves(training_losses, training_accs, fold_dir)
        
        # Guardar checkpoint del fold
        checkpoint = {
            'fold': fold,
            'model_state_dict': model.state_dict(),
            'test_patient': test_patient_id,
            'optimal_threshold': optimal_threshold,
            'metrics': fold_results,
            'class_weights': class_weights
        }
        torch.save(checkpoint, os.path.join(fold_dir, 'model.pth'))
    
    # ==========================================
    # RESULTADOS AGREGADOS
    # ==========================================
    print(f"\n{'='*70}")
    print("📊 RESULTATS AGREGATS K-FOLD LOPO")
    print(f"{'='*70}")
    
    results_df = pd.DataFrame(all_results)
    
    print(f"\n📋 RESULTADOS POR PACIENTE:")
    print(results_df[['test_patient', 'optimal_threshold', 'accuracy_opt', 
                      'f1_opt', 'sensitivity_opt', 'specificity_opt']].to_string(index=False))
    
    print(f"\n📈 ESTADÍSTICAS PROMEDIO (±std):")
    print(f"  Accuracy:      {results_df['accuracy_opt'].mean():.4f} ± {results_df['accuracy_opt'].std():.4f}")
    print(f"  F1-Score:      {results_df['f1_opt'].mean():.4f} ± {results_df['f1_opt'].std():.4f}")
    print(f"  Sensibilitat:  {results_df['sensitivity_opt'].mean():.4f} ± {results_df['sensitivity_opt'].std():.4f} ⭐")
    print(f"  Especificitat: {results_df['specificity_opt'].mean():.4f} ± {results_df['specificity_opt'].std():.4f}")
    print(f"  AUC:           {results_df['auc'].mean():.4f} ± {results_df['auc'].std():.4f}")
    
    # Comparación threshold default vs optimizado
    print(f"\n📊 MEJORA CON THRESHOLD OPTIMIZADO:")
    print(f"  Sensibilidad default: {results_df['sensitivity_default'].mean():.4f}")
    print(f"  Sensibilidad optimiz: {results_df['sensitivity_opt'].mean():.4f}")
    print(f"  Mejora: +{(results_df['sensitivity_opt'].mean() - results_df['sensitivity_default'].mean())*100:.1f}%")
    
    # Guardar resultados
    results_df.to_csv(os.path.join(results_dir, 'kfold_results.csv'), index=False)
    
    with open(os.path.join(results_dir, 'summary.txt'), 'w') as f:
        f.write("K-FOLD LOPO RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Number of folds: {len(test_patients_ids)}\n")
        f.write(f"Test patients: {', '.join(test_patients_ids)}\n\n")
        f.write("MEAN METRICS:\n")
        f.write(f"  Accuracy:      {results_df['accuracy_opt'].mean():.4f} ± {results_df['accuracy_opt'].std():.4f}\n")
        f.write(f"  F1-Score:      {results_df['f1_opt'].mean():.4f} ± {results_df['f1_opt'].std():.4f}\n")
        f.write(f"  Sensitivity:   {results_df['sensitivity_opt'].mean():.4f} ± {results_df['sensitivity_opt'].std():.4f}\n")
        f.write(f"  Specificity:   {results_df['specificity_opt'].mean():.4f} ± {results_df['specificity_opt'].std():.4f}\n")
        f.write(f"  AUC:           {results_df['auc'].mean():.4f} ± {results_df['auc'].std():.4f}\n")
    
    print(f"\n💾 Resultats desats a: {results_dir}\n")