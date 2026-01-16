import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def plot_patient_distribution(patients_df, save_dir):
    """Visualitza distribució d'horas i crisis per pacient"""
    ensure_dir(save_dir)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Hores per pacient
    ax1 = axes[0]
    patients_df = patients_df.sort_values('hours', ascending=False)
    colors = ['red' if not p else 'steelblue' for p in patients_df['is_pediatric']]
    ax1.bar(range(len(patients_df)), patients_df['hours'], color=colors)
    ax1.set_xlabel('Pacientes')
    ax1.set_ylabel('Horas de EEG')
    ax1.set_title('Distribución de horas de EEG por paciente')
    ax1.set_xticks(range(len(patients_df)))
    ax1.set_xticklabels(patients_df['patient_id'], rotation=45, ha='right')
    ax1.legend(['Pediàtric', 'Adult'])
    ax1.grid(axis='y', alpha=0.3)
    
    # Minuts de crisis per pacient
    ax2 = axes[1]
    patients_df = patients_df.sort_values('seizure_minutes', ascending=False)
    colors = ['red' if not p else 'coral' for p in patients_df['is_pediatric']]
    ax2.bar(range(len(patients_df)), patients_df['seizure_minutes'], color=colors)
    ax2.set_xlabel('Pacientes')
    ax2.set_ylabel('Minutos de crisis')
    ax2.set_title('Distribución de minutos de crisis por paciente')
    ax2.set_xticks(range(len(patients_df)))
    ax2.set_xticklabels(patients_df['patient_id'], rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'patient_distribution.png'), dpi=150)
    plt.close()

def save_roc(fpr, tpr, auc_score, save_dir):
    ensure_dir(save_dir)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "roc_curve.png"))
    plt.close()

def save_precision_recall(precision, recall, save_dir):
    ensure_dir(save_dir)
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid()
    plt.savefig(os.path.join(save_dir, "precision_recall_curve.png"))
    plt.close()

def save_confusion_matrix(cm, save_dir):
    ensure_dir(save_dir)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

def save_training_curves(losses, accuracies, save_dir):
    ensure_dir(save_dir)

    # Loss curve
    plt.figure(figsize=(6,5))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid()
    plt.savefig(os.path.join(save_dir, "training_loss.png"))
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(6,5))
    plt.plot(accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.grid()
    plt.savefig(os.path.join(save_dir, "training_accuracy.png"))
    plt.close()
