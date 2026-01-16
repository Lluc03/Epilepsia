import numpy as np
import pandas as pd
import glob
import os
from collections import Counter
from sklearn.model_selection import train_test_split

def load_patients_metadata(dataset_path):
    """Carga metadatos de todos los pacientes"""
    npz_files = sorted(glob.glob(os.path.join(dataset_path, "*_EEGwindow_*.npz")))
    
    patients_data = []
    for npz_path in npz_files:
        pat_id = os.path.basename(npz_path).split("_")[0]
        
        # Cargar metadata
        meta_candidates = glob.glob(os.path.join(dataset_path, f"{pat_id}*metadata*.parquet"))
        if len(meta_candidates) == 0:
            continue
            
        meta_path = meta_candidates[0]
        meta = pd.read_parquet(meta_path)
        
        # Contar clases
        class_counts = Counter(meta["class"].values)
        
        # Calcular horas de grabación (1 ventana = 1 segundo)
        hours = len(meta) / 3600
        seizure_minutes = class_counts[1] / 60
        
        # Identificar si es pediátrico (chb24 es el adulto)
        is_pediatric = pat_id != 'chb24'
        
        patients_data.append({
            'patient_id': pat_id,
            'npz_path': npz_path,
            'meta_path': meta_path,
            'n_samples': len(meta),
            'n_seizure': class_counts[1],
            'n_normal': class_counts[0],
            'hours': hours,
            'seizure_minutes': seizure_minutes,
            'is_pediatric': is_pediatric
        })
    
    df = pd.DataFrame(patients_data)
    
    # Estadísticas globales
    total_hours = df['hours'].sum()
    total_seizure_minutes = df['seizure_minutes'].sum()
    n_pediatric = df['is_pediatric'].sum()
    
    print(f"\n{'='*70}")
    print(f"📊 ANÁLISIS DEL DATASET CHB-MIT")
    print(f"{'='*70}")
    print(f"  Total pacientes:       {len(df)}")
    print(f"  Pediátricos:          {n_pediatric}")
    print(f"  Adultos:              {len(df) - n_pediatric}")
    print(f"\n  Total horas EEG:      {total_hours:.1f} h")
    print(f"  Total muestras:       {df['n_samples'].sum():,}")
    print(f"  Crisis totales:       {df['n_seizure'].sum():,} ventanas ({total_seizure_minutes:.1f} min)")
    print(f"  Ratio ictal/total:    {df['n_seizure'].sum() / df['n_samples'].sum():.3%}")
    print(f"{'='*70}\n")
    
    return df

def select_test_patient(patients_df, patient_id=None):
    """
    Selecciona paciente pediátrico para test
    Si patient_id es None, elige el que tenga más crisis
    """
    # Solo pacientes pediátricos
    pediatric = patients_df[patients_df['is_pediatric'] == True].copy()
    
    if patient_id is None:
        # Seleccionar el que tiene más crisis
        test_patient = pediatric.nlargest(1, 'n_seizure')
        patient_id = test_patient['patient_id'].values[0]
    else:
        test_patient = pediatric[pediatric['patient_id'] == patient_id]
        if len(test_patient) == 0:
            raise ValueError(f"Paciente {patient_id} no encontrado o no es pediátrico")
    
    train_patients = patients_df[patients_df['patient_id'] != patient_id].copy()
    
    print(f"🔄 LOPO Split (Leave-One-Patient-Out):")
    print(f"  📘 TRAIN: {len(train_patients)} pacientes")
    print(f"     - Total horas: {train_patients['hours'].sum():.1f} h")
    print(f"     - Crisis: {train_patients['seizure_minutes'].sum():.1f} min")
    print(f"\n  📕 TEST: {patient_id}")
    print(f"     - Horas: {test_patient['hours'].values[0]:.1f} h")
    print(f"     - Crisis: {test_patient['seizure_minutes'].values[0]:.1f} min")
    print(f"     - Muestras: {test_patient['n_samples'].values[0]:,}")
    
    return train_patients, test_patient

def load_patient_data(npz_path, meta_path):
    """Carga datos de un paciente específico"""
    npz = np.load(npz_path, allow_pickle=True)
    X = np.array(npz["EEG_win"], dtype=object)
    if X.dtype == object:
        X = np.stack(X, axis=0)
    
    meta = pd.read_parquet(meta_path)
    y = meta["class"].values
    
    return X.astype(np.float32), y.astype(np.int64)

def balance_train_data(X, y, strategy='undersample', target_hours=40):
    """
    Balancea datos de entrenamiento
    target_hours: horas aproximadas de datos balanceados (default 40h)
    """
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    
    n_normal = len(idx_0)
    n_seizure = len(idx_1)
    
    print(f"\n{'='*70}")
    print(f"⚖️  BALANCEO DE DATOS DE ENTRENAMIENTO")
    print(f"{'='*70}")
    print(f"  📊 ANTES del balanceo:")
    print(f"     - Normal (0):    {n_normal:8,} ({n_normal/3600:.1f}h) | {n_normal/(n_normal+n_seizure)*100:.1f}%")
    print(f"     - Epilepsia (1): {n_seizure:8,} ({n_seizure/60:.1f}min) | {n_seizure/(n_normal+n_seizure)*100:.1f}%")
    print(f"     - Total:         {n_normal+n_seizure:8,} ({(n_normal+n_seizure)/3600:.1f}h)")
    
    # Calcular tamaño objetivo (en ventanas)
    target_samples = int(target_hours * 3600)  # horas → segundos → ventanas
    
    if strategy == 'undersample':
        # Reducir normal a 2x seizure, pero limitado por target_hours
        n_target_per_class = min(n_seizure, target_samples // 2)
        idx_0_sampled = np.random.choice(idx_0, n_target_per_class, replace=False)
        idx_1_sampled = idx_1  # Todas las crisis
        
    elif strategy == 'mixed':
        # Balance 50-50 con límite de horas
        n_target_per_class = min(n_seizure, target_samples // 2)
        idx_0_sampled = np.random.choice(idx_0, n_target_per_class, replace=False)
        idx_1_sampled = np.random.choice(idx_1, n_target_per_class, replace=True)  # oversample seizure
    
    idx_balanced = np.concatenate([idx_0_sampled, idx_1_sampled])
    np.random.shuffle(idx_balanced)
    
    X_balanced = X[idx_balanced]
    y_balanced = y[idx_balanced]
    
    n_final_0 = np.sum(y_balanced == 0)
    n_final_1 = np.sum(y_balanced == 1)
    
    print(f"\n  📊 DESPUÉS del balanceo ({strategy}):")
    print(f"     - Normal (0):    {n_final_0:8,} ({n_final_0/3600:.1f}h) | {n_final_0/len(y_balanced)*100:.1f}%")
    print(f"     - Epilepsia (1): {n_final_1:8,} ({n_final_1/60:.1f}min) | {n_final_1/len(y_balanced)*100:.1f}%")
    print(f"     - Total:         {len(y_balanced):8,} ({len(y_balanced)/3600:.1f}h)")
    print(f"{'='*70}\n")
    
    return X_balanced, y_balanced

def temporal_train_val_split(X, y, val_ratio=0.15):
    """
    Split temporal (sin shuffle) para validation
    Toma los últimos val_ratio% para validation
    """
    n_total = len(X)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    
    # Split temporal: primeros para train, últimos para val
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:]
    y_val = y[n_train:]
    
    print(f"🔪 Split temporal (shuffle=False, val_ratio={val_ratio}):")
    print(f"   Train: ventanas [0 : {n_train}]")
    print(f"   Val:   ventanas [{n_train} : {n_total}]")
    
    return X_train, X_val, y_train, y_val

def load_lopo_data(train_patients, test_patient, balance_strategy='mixed', 
                   target_hours=40, val_ratio=0.15):
    """
    Carga datos con LOPO split y shuffle=False
    """
    print(f"\n{'='*70}")
    print(f"📂 CARGANDO DATOS DE PACIENTES")
    print(f"{'='*70}")
    
    # Cargar TRAIN (todos los pacientes menos test)
    X_train_list, y_train_list = [], []
    for _, row in train_patients.iterrows():
        X, y = load_patient_data(row['npz_path'], row['meta_path'])
        X_train_list.append(X)
        y_train_list.append(y)
        print(f"  ✓ {row['patient_id']}: {len(X):7,} ventanas ({len(X)/3600:.1f}h) "
              f"| Seizure: {np.sum(y==1):5,} ({np.sum(y==1)/60:.1f}min)")
    
    X_train = np.concatenate(X_train_list)
    y_train = np.concatenate(y_train_list)
    
    # Balancear TRAIN
    X_train, y_train = balance_train_data(X_train, y_train, 
                                          strategy=balance_strategy,
                                          target_hours=target_hours)
    
    # Split temporal Train/Val (shuffle=False)
    X_train, X_val, y_train, y_val = temporal_train_val_split(X_train, y_train, val_ratio)
    
    # Cargar TEST
    print(f"\n📕 Cargando TEST:")
    X_test, y_test = load_patient_data(
        test_patient['npz_path'].values[0],
        test_patient['meta_path'].values[0]
    )
    print(f"  ✓ {test_patient['patient_id'].values[0]}: {len(X_test):7,} ventanas "
          f"| Seizure: {np.sum(y_test==1):5,}")
    
    # Resumen final
    print(f"\n{'='*70}")
    print(f"📦 DATOS FINALES PARA ENTRENAMIENTO")
    print(f"{'='*70}")
    
    print(f"\n  🔵 TRAIN: {X_train.shape}")
    print(f"     - Normal (0):    {np.sum(y_train==0):8,} ({np.sum(y_train==0)/3600:.1f}h) | {np.sum(y_train==0)/len(y_train)*100:.1f}%")
    print(f"     - Epilepsia (1): {np.sum(y_train==1):8,} ({np.sum(y_train==1)/60:.1f}min) | {np.sum(y_train==1)/len(y_train)*100:.1f}%")
    
    print(f"\n  🟢 VALIDATION: {X_val.shape}")
    print(f"     - Normal (0):    {np.sum(y_val==0):8,} | {np.sum(y_val==0)/len(y_val)*100:.1f}%")
    print(f"     - Epilepsia (1): {np.sum(y_val==1):8,} | {np.sum(y_val==1)/len(y_val)*100:.1f}%")
    
    print(f"\n  🔴 TEST (paciente no visto): {X_test.shape}")
    print(f"     - Normal (0):    {np.sum(y_test==0):8,} | {np.sum(y_test==0)/len(y_test)*100:.1f}%")
    print(f"     - Epilepsia (1): {np.sum(y_test==1):8,} | {np.sum(y_test==1)/len(y_test)*100:.1f}%")
    print(f"{'='*70}\n")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)