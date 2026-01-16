import numpy as np
import pandas as pd
import glob
import os
from collections import Counter


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


def load_patient_data(npz_path, meta_path):
    """Carga datos de un paciente específico PRESERVANDO EL ORDEN TEMPORAL"""
    npz = np.load(npz_path, allow_pickle=True)
    X = np.array(npz["EEG_win"], dtype=object)
    if X.dtype == object:
        X = np.stack(X, axis=0)
    
    meta = pd.read_parquet(meta_path)
    y = meta["class"].values
    
    return X.astype(np.float32), y.astype(np.int64)


def temporal_balance_train_data(X, y, target_ratio=0.3, respect_temporality=True, 
                                include_transitions=True, transition_window=10):
    """
    Balancea datos INCLUYENDO transiciones entre estados
    
    Estrategia balanceada para LSTM en clasificación:
    1. Identifica segmentos contiguos de normal y seizure
    2. PRIORIZA (sin sobreenfatizar) bloques que contienen transiciones
    3. Completa con bloques contiguos normales adicionales
    4. Mantiene TODO el periodo ictal (seizure) completo
    5. NO hace shuffle - preserva continuidad temporal
    
    Args:
        X: datos [n_samples, 21, 128]
        y: etiquetas [n_samples]
        target_ratio: ratio objetivo de seizure/total (default 0.3)
        respect_temporality: si True, conserva secuencias continuas
        include_transitions: si True, asegura capturar transiciones
        transition_window: ventanas extra alrededor de transiciones (default 10s)
    
    Returns:
        X_balanced, y_balanced (Secuencias CONTINUAS incluyendo transiciones)
    """
    
    idx_normal = np.where(y == 0)[0]
    idx_seizure = np.where(y == 1)[0]
    
    n_normal = len(idx_normal)
    n_seizure = len(idx_seizure)
    
    print(f"\n{'='*70}")
    print(f"⚖️  BALANCEO CON BLOQUES CONTIGUOS + TRANSICIONES")
    print(f"{'='*70}")
    print(f"  📊 ANTES del balanceo:")
    print(f"     - Normal (0):    {n_normal:8,} ({n_normal/3600:.1f}h) | {n_normal/(n_normal+n_seizure)*100:.1f}%")
    print(f"     - Epilepsia (1): {n_seizure:8,} ({n_seizure/60:.1f}min) | {n_seizure/(n_normal+n_seizure)*100:.1f}%")
    
    # Calcular cuántos normales necesitamos
    n_normal_target = int(n_seizure * (1 - target_ratio) / target_ratio)
    n_normal_target = min(n_normal_target, n_normal)
    
    if respect_temporality and n_normal_target < n_normal:
        
        # PASO 1: IDENTIFICAR SEGMENTOS CONTIGUOS DE CLASE NORMAL
        normal_diffs = np.diff(idx_normal)
        segment_breaks = np.where(normal_diffs > 1)[0] + 1
        
        segments = []
        start = 0
        for break_point in segment_breaks:
            segments.append({
                'indices': idx_normal[start:break_point],
                'start': idx_normal[start],
                'end': idx_normal[break_point - 1],
                'length': break_point - start
            })
            start = break_point
        segments.append({
            'indices': idx_normal[start:],
            'start': idx_normal[start],
            'end': idx_normal[-1],
            'length': len(idx_normal) - start
        })
        
        print(f"\n  🔍 Análisis de segmentos normales:")
        print(f"     - Segmentos contiguos: {len(segments)}")
        segment_lengths = [s['length'] for s in segments]
        print(f"     - Longitud promedio: {np.mean(segment_lengths):.1f} ventanas")
        
        # PASO 2: IDENTIFICAR TRANSICIONES (sin sobreenfatizar)
        if include_transitions:
            transitions = np.diff(y.astype(int))
            idx_start_seizure = np.where(transitions == 1)[0] + 1   # 0→1
            idx_end_seizure = np.where(transitions == -1)[0] + 1    # 1→0
            
            print(f"\n  🔄 Transiciones detectadas:")
            print(f"     - Inicios de crisis (0→1): {len(idx_start_seizure)}")
            print(f"     - Finales de crisis (1→0): {len(idx_end_seizure)}")
            
            # Marcar segmentos que contienen o están cerca de transiciones
            for seg in segments:
                seg['has_transition'] = False
                seg['transition_priority'] = 0
                
                # Verificar si el segmento está cerca de alguna transición
                for trans_idx in np.concatenate([idx_start_seizure, idx_end_seizure]):
                    # Si la transición está dentro o cerca (±transition_window) del segmento
                    if (seg['start'] - transition_window <= trans_idx <= seg['end'] + transition_window):
                        seg['has_transition'] = True
                        # Calcular prioridad: más alta cuanto más cerca está la transición
                        distance = min(abs(seg['start'] - trans_idx), abs(seg['end'] - trans_idx))
                        seg['transition_priority'] = max(seg['transition_priority'], 1.0 / (1.0 + distance))
            
            n_segments_with_transitions = sum(1 for s in segments if s['has_transition'])
            print(f"     - Segmentos con/cerca transiciones: {n_segments_with_transitions}/{len(segments)}")
        
        # PASO 3: SELECCIÓN DE SEGMENTOS
        # Estrategia: Tomar segmentos con transiciones primero, luego rellenar con otros
        
        if include_transitions:
            # Ordenar: primero los que tienen transiciones, luego por longitud
            segments_sorted = sorted(segments, 
                                    key=lambda s: (s['has_transition'], s['transition_priority'], s['length']), 
                                    reverse=True)
        else:
            # Solo ordenar por longitud (segmentos más largos primero para continuidad)
            segments_sorted = sorted(segments, key=lambda s: s['length'], reverse=True)
        
        selected_segments = []
        current_count = 0
        n_transition_segments = 0
        
        for segment in segments_sorted:
            if current_count >= n_normal_target:
                break
            
            if current_count + segment['length'] <= n_normal_target:
                # Tomar el segmento completo
                selected_segments.append(segment)
                current_count += segment['length']
                if segment.get('has_transition', False):
                    n_transition_segments += 1
            else:
                # Tomar parte del segmento para completar
                remaining = n_normal_target - current_count
                if remaining > 0:
                    # Si tiene transición, intentar capturarla
                    if segment.get('has_transition', False):
                        # Tomar desde el inicio del segmento
                        partial_indices = segment['indices'][:remaining]
                    else:
                        # Tomar desde el inicio
                        partial_indices = segment['indices'][:remaining]
                    
                    partial_segment = {
                        'indices': partial_indices,
                        'start': partial_indices[0],
                        'end': partial_indices[-1],
                        'length': len(partial_indices),
                        'has_transition': segment.get('has_transition', False)
                    }
                    selected_segments.append(partial_segment)
                    current_count += remaining
                break
        
        # Concatenar índices de segmentos seleccionados
        idx_normal_sampled = np.concatenate([seg['indices'] for seg in selected_segments])
        
        print(f"\n  ✅ Segmentos seleccionados:")
        print(f"     - Total segmentos: {len(selected_segments)}")
        if include_transitions:
            print(f"     - Con transiciones: {n_transition_segments}")
            print(f"     - Sin transiciones: {len(selected_segments) - n_transition_segments}")
        print(f"     - Ventanas totales: {len(idx_normal_sampled):,}")
        
    else:
        idx_normal_sampled = idx_normal
    
    # PASO 4: COMBINAR CON SEIZURES Y ORDENAR
    idx_combined = np.concatenate([idx_normal_sampled, idx_seizure])
    idx_combined = np.sort(idx_combined)  # Mantener orden temporal
    
    X_balanced = X[idx_combined]
    y_balanced = y[idx_combined]
    
    n_final_0 = np.sum(y_balanced == 0)
    n_final_1 = np.sum(y_balanced == 1)
    
    print(f"\n  📊 DESPUÉS del balanceo:")
    print(f"     - Normal (0):    {n_final_0:8,} ({n_final_0/3600:.1f}h) | {n_final_0/len(y_balanced)*100:.1f}%")
    print(f"     - Epilepsia (1): {n_final_1:8,} ({n_final_1/60:.1f}min) | {n_final_1/len(y_balanced)*100:.1f}%")
    print(f"     - Total:         {len(y_balanced):8,} ({len(y_balanced)/3600:.1f}h)")
    
    # Análisis de continuidad
    all_diffs = np.diff(idx_combined)
    n_gaps = np.sum(all_diffs > 1)
    continuous_blocks = n_gaps + 1
    
    # Verificar que capturamos transiciones
    if include_transitions:
        transitions_in_data = np.sum(np.abs(np.diff(y_balanced)) == 1)
        print(f"\n  🔗 Continuidad y transiciones:")
        print(f"     - Bloques continuos: {continuous_blocks}")
        print(f"     - Gaps en secuencia: {n_gaps}")
        print(f"     - Transiciones capturadas: {transitions_in_data}")
    else:
        print(f"\n  🔗 Continuidad:")
        print(f"     - Bloques continuos: {continuous_blocks}")
        print(f"     - Gaps en secuencia: {n_gaps}")
    
    print(f"{'='*70}\n")
    
    return X_balanced, y_balanced


def temporal_train_val_split(X, y, val_ratio=0.15):
    """
    Split temporal estricto para validation
    
    IMPORTANTE: 
    - Train: primeros (1-val_ratio)% de datos
    - Val: últimos val_ratio% de datos
    - SIN SHUFFLE para respetar temporalidad
    
    Esto simula un escenario realista donde validamos en datos futuros
    """
    n_total = len(X)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    
    # Split temporal: primeros para train, últimos para val
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:]
    y_val = y[n_train:]
    
    print(f"\n🔪 Split temporal ESTRICTO (val_ratio={val_ratio}):")
    print(f"   Train: ventanas [0 : {n_train}] (primeras en el tiempo)")
    print(f"   Val:   ventanas [{n_train} : {n_total}] (últimas en el tiempo)")
    print(f"   ⚠️  SIN SHUFFLE - orden temporal respetado\n")
    
    return X_train, X_val, y_train, y_val


def load_lopo_data_temporal(train_patients, test_patient, target_ratio=0.3, val_ratio=0.15,
                           include_transitions=True, transition_window=10):
    """
    Carga datos con LOPO split RESPETANDO TEMPORALIDAD
    
    Cambios clave vs versión anterior:
    1. NO se hace shuffle en ningún momento
    2. Balance temporal con bloques contiguos
    3. INCLUYE transiciones entre estados (sin sobreenfatizar)
    4. Split temporal estricto train/val
    """
    print(f"\n{'='*70}")
    print(f"📂 CARGANDO DATOS - LOPO TEMPORAL CON TRANSICIONES")
    print(f"{'='*70}")
    
    # Cargar TRAIN (todos los pacientes menos test)
    X_train_list, y_train_list = [], []
    for _, row in train_patients.iterrows():
        X, y = load_patient_data(row['npz_path'], row['meta_path'])
        X_train_list.append(X)
        y_train_list.append(y)
        print(f"  ✓ {row['patient_id']}: {len(X):7,} ventanas ({len(X)/3600:.1f}h) "
              f"| Seizure: {np.sum(y==1):5,} ({np.sum(y==1)/60:.1f}min)")
    
    # Concatenar manteniendo orden
    X_train = np.concatenate(X_train_list)
    y_train = np.concatenate(y_train_list)
    
    print(f"\n  📊 TRAIN concatenado: {X_train.shape}")
    print(f"     - Normal: {np.sum(y_train==0):,} | Seizure: {np.sum(y_train==1):,}")
    
    # Balancear RESPETANDO temporalidad e INCLUYENDO transiciones
    X_train, y_train = temporal_balance_train_data(
        X_train, y_train, 
        target_ratio=target_ratio,
        respect_temporality=True,
        include_transitions=include_transitions,
        transition_window=transition_window
    )
    
    # Split temporal Train/Val (SIN shuffle)
    X_train, X_val, y_train, y_val = temporal_train_val_split(
        X_train, y_train, val_ratio=val_ratio
    )
    
    # Cargar TEST
    print(f"📕 Cargando TEST:")
    X_test, y_test = load_patient_data(
        test_patient['npz_path'].values[0],
        test_patient['meta_path'].values[0]
    )
    print(f"  ✓ {test_patient['patient_id'].values[0]}: {len(X_test):7,} ventanas "
          f"| Seizure: {np.sum(y_test==1):5,}\n")
    
    # Resumen final
    print(f"{'='*70}")
    print(f"📦 DATOS FINALES (ORDEN TEMPORAL + TRANSICIONES)")
    print(f"{'='*70}")
    
    print(f"\n  🔵 TRAIN: {X_train.shape}")
    print(f"     - Normal (0):    {np.sum(y_train==0):8,} | {np.sum(y_train==0)/len(y_train)*100:.1f}%")
    print(f"     - Epilepsia (1): {np.sum(y_train==1):8,} | {np.sum(y_train==1)/len(y_train)*100:.1f}%")
    
    print(f"\n  🟢 VALIDATION: {X_val.shape}")
    print(f"     - Normal (0):    {np.sum(y_val==0):8,} | {np.sum(y_val==0)/len(y_val)*100:.1f}%")
    print(f"     - Epilepsia (1): {np.sum(y_val==1):8,} | {np.sum(y_val==1)/len(y_val)*100:.1f}%")
    
    print(f"\n  🔴 TEST (paciente no visto): {X_test.shape}")
    print(f"     - Normal (0):    {np.sum(y_test==0):8,} | {np.sum(y_test==0)/len(y_test)*100:.1f}%")
    print(f"     - Epilepsia (1): {np.sum(y_test==1):8,} | {np.sum(y_test==1)/len(y_test)*100:.1f}%")
    print(f"{'='*70}\n")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)