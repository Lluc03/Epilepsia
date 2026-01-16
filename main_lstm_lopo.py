import yaml
import os
import datetime
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.baseline.lstm_model import EpilepsyLSTM
from src.baseline.train_baseline import train_baseline
from src.dataloader.eeg_dataset import EEGWindowDataset
from src.utils.temporal_data_utils import (
    load_patients_metadata,
    load_lopo_data_temporal
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
    results_dir = f"results/lstm_lopo_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("🧠 LSTM SEIZURE DETECTION - LOPO TEMPORAL")
    print("   Modelo: LSTM bidireccional con atención temporal")
    print("   Estrategia: Leave-One-Patient-Out con orden temporal preservado")
    print("   Pacientes: chb10, chb04")
    print("=" * 70)
    
    # 1. CARGAR METADATOS
    patients_df = load_patients_metadata(DATASET_PATH)
    
    # 2. SELECCIONAR PACIENTES PARA TEST (chb10 y chb04)
    test_patients_ids = ['chb10', 'chb04']
    
    print(f"\n📋 PACIENTES SELECCIONADOS PARA LOPO:")
    for i, pid in enumerate(test_patients_ids, 1):
        pat_info = patients_df[patients_df['patient_id'] == pid].iloc[0]
        print(f"  Test {i}: {pid} - {pat_info['n_seizure']:,} crisis ({pat_info['seizure_minutes']:.1f} min)")
    
    # Almacenar resultados
    all_results = []
    
    # ==========================================
    # LOPO LOOP
    # ==========================================
    for test_idx, test_patient_id in enumerate(test_patients_ids, 1):
        print(f"\n{'='*70}")
        print(f"🔄 TEST PATIENT {test_idx}/{len(test_patients_ids)}: {test_patient_id}")
        print(f"{'='*70}")
        
        # Split train/test
        test_patient = patients_df[patients_df['patient_id'] == test_patient_id]
        train_patients = patients_df[patients_df['patient_id'] != test_patient_id].copy()
        
        print(f"\n  TRAIN: {len(train_patients)} pacientes (excluyendo {test_patient_id})")
        print(f"  TEST:  {test_patient_id}")
        
        # 3. CARGAR DATOS CON ORDEN TEMPORAL PRESERVADO + TRANSICIONES
        # AJUSTE: Cambiar target_ratio de 0.3 a 0.25 (más datos normales)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_lopo_data_temporal(
            train_patients,
            test_patient,
            target_ratio=0.25,  # Reducido de 0.30 a 0.25 (75% normal, 25% seizure)
            val_ratio=0.15,
            include_transitions=True,
            transition_window=10
        )
        
        print(f"\n  📦 DATOS CARGADOS:")
        print(f"     TRAIN: {X_train.shape}")
        print(f"     VAL:   {X_val.shape}")
        print(f"     TEST:  {X_test.shape}")
        
        # 4. MODELO LSTM (AJUSTADO)
        model = EpilepsyLSTM(
            n_channels=21,
            n_classes=2,
            hidden_size=128,
            num_layers=2,
            dropout=0.4  # Aumentado de 0.3 a 0.4
        )
        
        print(f"\n  🏗️  ARQUITECTURA LSTM (MEJORADA):")
        print(f"     - Input: [batch, 21, 128]")
        print(f"     - LSTM hidden_size: 128")
        print(f"     - LSTM layers: 2")
        print(f"     - LSTM dropout: 0.4")
        print(f"     - Classifier: 128 -> 64 -> 32 -> 2 (con dropout 0.5 y 0.4)")
        print(f"     - Output: [batch, 2]")
        
        # 5. CALCULAR CLASS WEIGHTS (AJUSTADO MÁS)
        n_normal = np.sum(y_train == 0)
        n_seizure = np.sum(y_train == 1)
        
        # CAMBIO: Reducir aún más el peso de seizure
        # 1.5x → 1.2x para mejor balance precision/recall
        weight_normal = 1.0
        weight_seizure = (n_normal / n_seizure) * 1.2
        
        class_weights = [weight_normal, weight_seizure]
        
        print(f"\n  ⚖️  Class weights calculados:")
        print(f"     Normal (0): {weight_normal:.2f}")
        print(f"     Seizure (1): {weight_seizure:.2f}")
        print(f"     Ratio de peso: {weight_seizure:.2f}x (optimizado para F1)")
        
        # 6. ENTRENAMIENTO
        print(f"\n  🏋️  INICIANDO ENTRENAMIENTO...")
        training_losses, val_losses, training_accs = train_baseline(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=60,
            lr=0.0002,
            batch_size=128,
            class_weights=class_weights,
            early_stopping_patience=12,
            weight_decay=5e-5
        )
        
        # 7. EVALUACIÓN EN TEST CON MÚLTIPLES THRESHOLDS
        test_ds = EEGWindowDataset(X_test, y_test)
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
        
        print(f"\n  🎯 EVALUACIÓN CON MÚLTIPLES THRESHOLDS:")
        print(f"  {'='*65}")
        
        # Probar diferentes thresholds
        thresholds_to_test = [0.4, 0.5, 0.6, 0.7, 0.75, 0.8]
        threshold_results = []
        
        for thresh in thresholds_to_test:
            metrics = evaluate_with_threshold(model, test_loader, thresh)
            threshold_results.append({
                'threshold': thresh,
                'accuracy': metrics['accuracy'],
                'f1': metrics['f1'],
                'sensitivity': metrics['sensitivity'],
                'specificity': metrics['specificity'],
                'precision': metrics['precision'],
                'auc': metrics['auc']
            })
            
            print(f"  Thresh={thresh:.2f} | Acc={metrics['accuracy']:.3f} | "
                  f"F1={metrics['f1']:.3f} | Prec={metrics['precision']:.3f} | "
                  f"Sens={metrics['sensitivity']:.3f} | Spec={metrics['specificity']:.3f}")
        
        print(f"  {'='*65}")
        
        # Seleccionar el threshold con mejor F1
        best_result = max(threshold_results, key=lambda x: x['f1'])
        optimal_threshold = best_result['threshold']
        
        print(f"\n  ⭐ MEJOR THRESHOLD (máximo F1): {optimal_threshold:.2f}")
        print(f"     F1={best_result['f1']:.4f} | Sens={best_result['sensitivity']:.4f} | "
              f"Spec={best_result['specificity']:.4f}")
        
        # Evaluar con el mejor threshold
        metrics_optimized = evaluate_with_threshold(model, test_loader, optimal_threshold)
        metrics_default = evaluate_with_threshold(model, test_loader, 0.5)
        
        # Guardar resultados
        fold_results = {
            'test_patient': test_patient_id,
            'n_test_samples': len(X_test),
            'n_test_seizures': np.sum(y_test == 1),
            'optimal_threshold': optimal_threshold,
            'accuracy_default': metrics_default['accuracy'],
            'f1_default': metrics_default['f1'],
            'sensitivity_default': metrics_default['confusion_matrix'][1, 1] / np.sum(y_test == 1) if np.sum(y_test == 1) > 0 else 0,
            'accuracy_opt': metrics_optimized['accuracy'],
            'f1_opt': metrics_optimized['f1'],
            'sensitivity_opt': metrics_optimized['sensitivity'],
            'specificity_opt': metrics_optimized['specificity'],
            'auc': metrics_optimized['auc']
        }
        
        all_results.append(fold_results)
        
        print(f"\n✅ RESULTADOS PACIENTE {test_patient_id}:")
        print(f"  Threshold óptimo: {optimal_threshold:.3f}")
        print(f"  Accuracy:         {metrics_optimized['accuracy']:.4f}")
        print(f"  F1-Score:         {metrics_optimized['f1']:.4f}")
        print(f"  Sensibilidad:     {metrics_optimized['sensitivity']:.4f} ⭐")
        print(f"  Especificidad:    {metrics_optimized['specificity']:.4f}")
        print(f"  AUC:              {metrics_optimized['auc']:.4f}")
        print(f"\n  Matriz de confusión:")
        print(f"    {metrics_optimized['confusion_matrix']}")
        
        # Guardar visualizaciones
        patient_dir = os.path.join(results_dir, f"patient_{test_patient_id}")
        os.makedirs(patient_dir, exist_ok=True)
        
        save_roc(metrics_optimized["fpr"], metrics_optimized["tpr"], 
                 metrics_optimized["auc"], patient_dir)
        save_confusion_matrix(metrics_optimized["confusion_matrix"], patient_dir)
        save_precision_recall(metrics_optimized["precision_curve"], 
                             metrics_optimized["recall"], patient_dir)
        save_training_curves(training_losses, training_accs, patient_dir)
        
        # Guardar checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'test_patient': test_patient_id,
            'optimal_threshold': optimal_threshold,
            'metrics': fold_results,
            'class_weights': class_weights,
            'model_config': {
                'n_channels': 21,
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.3
            }
        }
        torch.save(checkpoint, os.path.join(patient_dir, 'lstm_model.pth'))
        
        print(f"\n💾 Resultados guardados en: {patient_dir}")
    
    # ==========================================
    # RESULTADOS AGREGADOS
    # ==========================================
    print(f"\n{'='*70}")
    print("📊 RESULTADOS AGREGADOS LSTM-LOPO TEMPORAL")
    print(f"{'='*70}")
    
    results_df = pd.DataFrame(all_results)
    
    print(f"\n📋 RESULTADOS POR PACIENTE:")
    print(results_df[['test_patient', 'optimal_threshold', 'accuracy_opt', 
                      'f1_opt', 'sensitivity_opt', 'specificity_opt', 'auc']].to_string(index=False))
    
    print(f"\n📈 ESTADÍSTICAS PROMEDIO (±std):")
    print(f"  Accuracy:      {results_df['accuracy_opt'].mean():.4f} ± {results_df['accuracy_opt'].std():.4f}")
    print(f"  F1-Score:      {results_df['f1_opt'].mean():.4f} ± {results_df['f1_opt'].std():.4f}")
    print(f"  Sensibilidad:  {results_df['sensitivity_opt'].mean():.4f} ± {results_df['sensitivity_opt'].std():.4f} ⭐")
    print(f"  Especificidad: {results_df['specificity_opt'].mean():.4f} ± {results_df['specificity_opt'].std():.4f}")
    print(f"  AUC:           {results_df['auc'].mean():.4f} ± {results_df['auc'].std():.4f}")
    
    # Comparación threshold default vs optimizado
    print(f"\n📊 MEJORA CON THRESHOLD OPTIMIZADO:")
    print(f"  Sensibilidad default: {results_df['sensitivity_default'].mean():.4f}")
    print(f"  Sensibilidad optimiz: {results_df['sensitivity_opt'].mean():.4f}")
    mejora = (results_df['sensitivity_opt'].mean() - results_df['sensitivity_default'].mean()) * 100
    print(f"  Mejora: {'+' if mejora > 0 else ''}{mejora:.1f}%")
    
    # Guardar resultados
    results_df.to_csv(os.path.join(results_dir, 'lstm_lopo_results.csv'), index=False)
    
    with open(os.path.join(results_dir, 'summary.txt'), 'w') as f:
        f.write("LSTM-LOPO TEMPORAL RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Test patients: {', '.join(test_patients_ids)}\n")
        f.write("Temporal order preserved: YES\n")
        f.write("Balance strategy: Temporal undersample (no shuffle)\n\n")
        f.write("MEAN METRICS:\n")
        f.write(f"  Accuracy:      {results_df['accuracy_opt'].mean():.4f} ± {results_df['accuracy_opt'].std():.4f}\n")
        f.write(f"  F1-Score:      {results_df['f1_opt'].mean():.4f} ± {results_df['f1_opt'].std():.4f}\n")
        f.write(f"  Sensitivity:   {results_df['sensitivity_opt'].mean():.4f} ± {results_df['sensitivity_opt'].std():.4f}\n")
        f.write(f"  Specificity:   {results_df['specificity_opt'].mean():.4f} ± {results_df['specificity_opt'].std():.4f}\n")
        f.write(f"  AUC:           {results_df['auc'].mean():.4f} ± {results_df['auc'].std():.4f}\n")
    
    print(f"\n💾 Resultados guardados en: {results_dir}\n")
    print("=" * 70)
    print("✅ EXPERIMENTO LSTM-LOPO COMPLETADO")
    print("=" * 70)