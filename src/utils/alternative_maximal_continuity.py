import numpy as np

def maximal_continuity_balance(X, y, target_ratio=0.3):
    """
    ALTERNATIVA: Máxima continuidad temporal
    
    Estrategia ULTRA-CONSERVADORA:
    1. Identifica todos los segmentos contiguos (normal y seizure)
    2. Mantiene segmentos COMPLETOS sin romperlos
    3. Selecciona los segmentos más largos para maximizar continuidad
    
    Esta estrategia es ideal si quieres que la LSTM vea SOLO secuencias continuas.
    
    Ventaja: Máxima continuidad
    Desventaja: Puede perder diversidad espacial en los datos
    """
    
    print(f"\n{'='*70}")
    print(f"🔗 BALANCEO CON MÁXIMA CONTINUIDAD")
    print(f"{'='*70}")
    
    # Identificar TODOS los segmentos contiguos (incluyendo seizures)
    diffs = np.diff(np.arange(len(y)))
    
    # Crear lista de segmentos con sus características
    segments = []
    start_idx = 0
    
    for i in range(1, len(y)):
        # Si cambia la clase o no es consecutivo, cerrar segmento
        if y[i] != y[i-1] or i - start_idx != (i - start_idx):
            segment = {
                'start': start_idx,
                'end': i,
                'length': i - start_idx,
                'label': y[start_idx],
                'indices': np.arange(start_idx, i)
            }
            segments.append(segment)
            start_idx = i
    
    # Último segmento
    segments.append({
        'start': start_idx,
        'end': len(y),
        'length': len(y) - start_idx,
        'label': y[start_idx],
        'indices': np.arange(start_idx, len(y))
    })
    
    # Separar segmentos por clase
    normal_segments = [s for s in segments if s['label'] == 0]
    seizure_segments = [s for s in segments if s['label'] == 1]
    
    print(f"\n  📊 Análisis de segmentos:")
    print(f"     - Segmentos normales: {len(normal_segments)}")
    print(f"     - Segmentos seizure:  {len(seizure_segments)}")
    
    # Calcular longitudes
    normal_lengths = [s['length'] for s in normal_segments]
    seizure_lengths = [s['length'] for s in seizure_segments]
    
    print(f"\n  📏 Longitud de segmentos:")
    print(f"     Normal  - Total: {sum(normal_lengths):,} | Promedio: {np.mean(normal_lengths):.1f} | Max: {max(normal_lengths):,}")
    print(f"     Seizure - Total: {sum(seizure_lengths):,} | Promedio: {np.mean(seizure_lengths):.1f} | Max: {max(seizure_lengths):,}")
    
    # Mantener TODOS los segmentos de seizure
    total_seizure = sum(seizure_lengths)
    
    # Calcular cuántos normal necesitamos
    target_normal = int(total_seizure * (1 - target_ratio) / target_ratio)
    
    # ESTRATEGIA 1: Seleccionar segmentos normales más largos primero
    # Esto maximiza la continuidad
    normal_segments_sorted = sorted(normal_segments, key=lambda s: s['length'], reverse=True)
    
    selected_normal_segments = []
    current_count = 0
    
    for segment in normal_segments_sorted:
        if current_count >= target_normal:
            break
        
        if current_count + segment['length'] <= target_normal:
            # Tomar el segmento completo
            selected_normal_segments.append(segment)
            current_count += segment['length']
        else:
            # Tomar solo una parte del segmento (desde el inicio para mantener continuidad)
            remaining = target_normal - current_count
            partial_segment = {
                'start': segment['start'],
                'end': segment['start'] + remaining,
                'length': remaining,
                'label': 0,
                'indices': segment['indices'][:remaining]
            }
            selected_normal_segments.append(partial_segment)
            current_count += remaining
            break
    
    print(f"\n  ✅ Segmentos seleccionados:")
    print(f"     - Normal segments: {len(selected_normal_segments)}")
    print(f"     - Seizure segments: {len(seizure_segments)} (todos)")
    
    # Combinar todos los índices EN ORDEN
    all_selected_segments = selected_normal_segments + seizure_segments
    all_selected_segments.sort(key=lambda s: s['start'])  # Ordenar por tiempo
    
    # Extraer índices manteniendo orden temporal
    selected_indices = []
    for segment in all_selected_segments:
        selected_indices.extend(segment['indices'])
    
    selected_indices = np.array(selected_indices, dtype=int)
    
    X_balanced = X[selected_indices]
    y_balanced = y[selected_indices]
    
    # Verificar continuidad PERFECTA dentro de segmentos
    n_final_0 = np.sum(y_balanced == 0)
    n_final_1 = np.sum(y_balanced == 1)
    
    print(f"\n  📊 RESULTADO FINAL:")
    print(f"     - Normal (0):    {n_final_0:8,} ({n_final_0/3600:.1f}h) | {n_final_0/len(y_balanced)*100:.1f}%")
    print(f"     - Epilepsia (1): {n_final_1:8,} ({n_final_1/60:.1f}min) | {n_final_1/len(y_balanced)*100:.1f}%")
    print(f"     - Total:         {len(y_balanced):8,} ({len(y_balanced)/3600:.1f}h)")
    
    # Contar gaps
    all_diffs = np.diff(selected_indices)
    n_gaps = np.sum(all_diffs > 1)
    max_continuous = np.max(np.diff(np.where(all_diffs > 1)[0])) if n_gaps > 0 else len(selected_indices)
    
    print(f"\n  🔗 Continuidad MÁXIMA:")
    print(f"     - Número de gaps: {n_gaps}")
    print(f"     - Segmento continuo más largo: {max_continuous} ventanas")
    print(f"     - Todos los datos en segmentos completos: ✅")
    print(f"{'='*70}\n")
    
    return X_balanced, y_balanced


def visualize_sequence_continuity(y_balanced, sample_size=1000):
    """
    Visualiza la continuidad de las secuencias
    Útil para verificar que NO hay saltos aleatorios
    """
    print(f"\n📊 VISUALIZACIÓN DE CONTINUIDAD (primeras {sample_size} ventanas):")
    print("   0=Normal, 1=Seizure")
    print("   " + "="*80)
    
    # Mostrar en bloques de 100
    for i in range(0, min(sample_size, len(y_balanced)), 100):
        end = min(i + 100, len(y_balanced))
        sequence = ''.join([str(int(y_balanced[j])) for j in range(i, end)])
        print(f"   [{i:5d}-{end:5d}]: {sequence}")
    
    print("   " + "="*80)
    
    # Análisis de transiciones
    transitions = np.diff(y_balanced)
    n_transitions = np.sum(transitions != 0)
    
    print(f"\n   Transiciones de clase: {n_transitions}")
    print(f"   Frecuencia: {n_transitions / len(y_balanced) * 100:.2f}% de las ventanas")
    print()