import yaml
import os
import datetime
import torch
from torch.utils.data import DataLoader

from src.baseline.cnn1d_separated import CNN1DSeparated
from src.baseline.train_baseline import train_baseline
from src.dataloader.eeg_dataset import EEGWindowDataset
from src.utils.data_utils import (
    load_patients_metadata,
    select_test_patient,
    load_lopo_data
)
from src.utils.metrics import find_optimal_threshold, evaluate_with_threshold, evaluate
from src.utils.visualizer import (
    save_roc, save_confusion_matrix,
    save_precision_recall, save_training_curves,
    plot_patient_distribution
)

if __name__ == "__main__":

    # --------------------------------------------------
    # CONFIGURACIÓ
    # --------------------------------------------------
    with open("src/config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    DATASET_PATH = config["dataset_path"]

    # Crear directori de resultats
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    results_dir = f"results/lopo_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("🧠 CLASSIFICADOR D’EPILÈPSIA – DATASET CHB-MIT")
    print("   Pipeline: Extractor de característiques → Classificador")
    print("   Estratègia: Leave-One-Patient-Out (LOPO)")
    print("=" * 70)

    # --------------------------------------------------
    # 1. CARREGAR METADADES DE TOTS ELS PACIENTS
    # --------------------------------------------------
    patients_df = load_patients_metadata(DATASET_PATH)

    # Visualitzar distribució de pacients
    plot_patient_distribution(patients_df, results_dir)

    # --------------------------------------------------
    # 2. SELECCIONAR PACIENT PEDIÀTRIC PER A TEST
    # --------------------------------------------------
    # Opció 1: Automàtica (el que té més crisis)
    train_patients, test_patient = select_test_patient(patients_df)

    # Opció 2: Manual (descomenta per escollir-ne un)
    # train_patients, test_patient = select_test_patient(patients_df, patient_id='chb01')

    # --------------------------------------------------
    # 3. CARREGAR DADES AMB ESTRATÈGIA LOPO
    # --------------------------------------------------
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_lopo_data(
        train_patients,
        test_patient,
        balance_strategy='mixed',   # 'undersample' o 'mixed'
        target_hours=40,            # Hores objectiu d’entrenament
        val_ratio=0.15              # 15% per a validació (temporal, sense shuffle)
    )

    # --------------------------------------------------
    # 4. MODEL AMB PIPELINE SEPARAT
    # --------------------------------------------------
    model = CNN1DSeparated(
        n_channels=config["model"]["n_channels"],
        n_classes=config["model"]["n_classes"]
    )

    print(f"\n{'=' * 70}")
    print("🏗️  ARQUITECTURA DEL MODEL")
    print(f"{'=' * 70}")
    print("  Extractor de característiques:")
    print(f"    - Paràmetres: {sum(p.numel() for p in model.feature_extractor.parameters()):,}")
    print("    - Sortida: Característiques [batch, 2]")
    print("\n  Classificador:")
    print(f"    - Paràmetres: {sum(p.numel() for p in model.classifier.parameters()):,}")
    print("    - Sortida: Logits [batch, 2]")
    print(f"\n  TOTAL: {sum(p.numel() for p in model.parameters()):,} paràmetres")
    print(f"{'=' * 70}\n")

    # --------------------------------------------------
    # 5. ENTRENAMENT
    # --------------------------------------------------
    print(f"{'=' * 70}")
    print("🚀 INICIANT ENTRENAMENT")
    print(f"{'=' * 70}")
    print(f"  Èpoques: {config['training']['epochs']}")
    print(f"  Mida del batch: {config['training']['batch_size']}")
    print(f"  Learning rate: {config['training']['lr']}")
    print(f"  Dispositiu: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"{'=' * 70}\n")

    training_losses, _, training_accuracies = train_baseline(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=config["training"]["epochs"],
        lr=config["training"]["lr"],
        batch_size=config["training"]["batch_size"]
    )

    # --------------------------------------------------
    # 6. AVALUACIÓ EN TEST (PACIENT NO VIST)
    # --------------------------------------------------
    print(f"\n{'=' * 70}")
    print("📊 AVALUACIÓ EN EL CONJUNT DE TEST (PACIENT NO VIST)")
    print(f"{'=' * 70}")
    
    test_ds = EEGWindowDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    
    # PASO 1: Evaluar con threshold por defecto (0.5)
    print("\n1️⃣ EVALUACIÓN CON THRESHOLD POR DEFECTO (0.5):")
    metrics_default = evaluate(model, test_loader)
    print(f"  Accuracy:       {metrics_default['accuracy']:.4f}")
    print(f"  F1-Score:       {metrics_default['f1']:.4f}")
    print(f"  AUC:            {metrics_default['auc']:.4f}")
    print(f"  Matriu de confusió:")
    print(f"    {metrics_default['confusion_matrix']}")
    
    # PASO 2: Encontrar threshold óptimo en VALIDACIÓN
    print("\n2️⃣ OPTIMIZANDO THRESHOLD EN VALIDACIÓN:")
    val_ds = EEGWindowDataset(X_val, y_val)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    
    # Opción A: Maximizar F1 con sensibilidad mínima del 70%
    optimal_threshold, _ = find_optimal_threshold(
        model, val_loader, 
        criterion='f1_min_sens',
        min_sensitivity=0.70
    )
    
    # Opción B: Maximizar solo sensibilidad (descomentar si prefieres esto)
#    optimal_threshold, _ = find_optimal_threshold(model, val_loader, criterion='sensitivity')
    
    # PASO 3: Evaluar en TEST con threshold óptimo
    print(f"\n3️⃣ EVALUACIÓN EN TEST CON THRESHOLD OPTIMIZADO ({optimal_threshold:.3f}):")
    metrics_optimized = evaluate_with_threshold(model, test_loader, optimal_threshold)
    
    print("\n✅ RESULTATS FINALS (THRESHOLD OPTIMITZAT):")
    print(f"  Pacient de test: {test_patient['patient_id'].values[0]}")
    print(f"  Threshold:      {metrics_optimized['threshold']:.3f}")
    print(f"  Accuracy:       {metrics_optimized['accuracy']:.4f}")
    print(f"  F1-Score:       {metrics_optimized['f1']:.4f}")
    print(f"  AUC:            {metrics_optimized['auc']:.4f}")
    print(f"  Sensibilidad:   {metrics_optimized['sensitivity']:.4f} ⭐")
    print(f"  Especificitat:  {metrics_optimized['specificity']:.4f}")
    print(f"\n  Matriu de confusió:")
    print(f"    {metrics_optimized['confusion_matrix']}")
    
    # Comparación antes/después
    tn_opt, fp_opt, fn_opt, tp_opt = metrics_optimized['confusion_matrix'].ravel()
    tn_def, fp_def, fn_def, tp_def = metrics_default['confusion_matrix'].ravel()
    
    print(f"\n📈 COMPARACIÓN THRESHOLD 0.5 vs {optimal_threshold:.3f}:")
    print(f"  Crisis detectadas:")
    print(f"    - Threshold 0.5:    {tp_def:5,} / {tp_def + fn_def:,} ({tp_def/(tp_def+fn_def)*100:.1f}%)")
    print(f"    - Threshold {optimal_threshold:.2f}: {tp_opt:5,} / {tp_opt + fn_opt:,} ({tp_opt/(tp_opt+fn_opt)*100:.1f}%)")
    print(f"  Falsos positius:")
    print(f"    - Threshold 0.5:    {fp_def:5,}")
    print(f"    - Threshold {optimal_threshold:.2f}: {fp_opt:5,}")
    
    # Usar métricas optimizadas para guardar
    metrics = metrics_optimized

    # --------------------------------------------------
    # 7. GUARDAR VISUALITZACIONS
    # --------------------------------------------------
    save_roc(metrics["fpr"], metrics["tpr"], metrics["auc"], results_dir)
    save_confusion_matrix(metrics["confusion_matrix"], results_dir)
    save_precision_recall(metrics["precision"], metrics["recall"], results_dir)
    save_training_curves(training_losses, training_accuracies, results_dir)

    # --------------------------------------------------
    # 8. GUARDAR MODEL I RESULTATS
    # --------------------------------------------------
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'test_patient': test_patient['patient_id'].values[0],
        'train_patients': list(train_patients['patient_id'].values),
        'config': config,
        'metrics': {
            'accuracy': float(metrics['accuracy']),
            'f1': float(metrics['f1']),
            'auc': float(metrics['auc'])
        },
        'confusion_matrix': metrics['confusion_matrix'].tolist(),
        'training_history': {
            'losses': training_losses,
            'accuracies': training_accuracies
        }
    }

    torch.save(checkpoint, os.path.join(results_dir, 'model_checkpoint.pth'))

    # Guardar resum en text
    with open(os.path.join(results_dir, 'summary.txt'), 'w') as f:
        f.write("CLASSIFICADOR D’EPILÈPSIA – CHB-MIT\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"Pacient TEST: {test_patient['patient_id'].values[0]}\n")
        f.write(f"Pacients TRAIN: {len(train_patients)}\n\n")
        f.write("RESULTATS:\n")
        f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"  F1-Score:  {metrics['f1']:.4f}\n")
        f.write(f"  AUC:       {metrics['auc']:.4f}\n\n")
        f.write(f"Matriu de confusió:\n{metrics['confusion_matrix']}\n")

    print(f"\n{'=' * 70}")
    print(f"💾 Resultats desats a: {results_dir}")
    print(f"{'=' * 70}\n")
