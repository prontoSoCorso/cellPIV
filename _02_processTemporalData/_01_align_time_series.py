import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.signal import savgol_filter
import warnings
import pickle

def method_pandas_resample(times, values, target_interval_minutes=15, total_points=96):
    """
    Metodo alternativo usando pandas resample - più robusto per dati molto irregolari
    """
    # Crea DataFrame con timestamp reali
    df = pd.DataFrame({
        'value': values,
        'timestamp': pd.to_datetime(times, unit='m')  # Converte minuti in datetime
    })
    df.set_index('timestamp', inplace=True)
    
    # Resample a intervalli regolari
    target_freq = f"{target_interval_minutes}min"
    resampled = df.resample(target_freq).mean()
    
    # Interpola valori mancanti
    resampled['value'] = resampled['value'].interpolate(method='time')
    
    # Taglia/estende alla lunghezza desiderata
    if len(resampled) > total_points:
        result = resampled.iloc[:total_points]['value'].values
    else:
        result = resampled['value'].values
        # Estendi con ultimo valore se necessario
        if len(result) < total_points:
            result = np.pad(result, (0, total_points - len(result)), 
                          mode='constant', constant_values=result[-1])
    
    return result

def method_spline_smooth(original_times, values, target_times, smooth_factor=0.1):
    """
    Interpolazione spline con smoothing - buona per dati rumorosi
    """
    if len(original_times) <= 3:
        return method_linear_fallback(original_times, values, target_times)
    
    try:
        # Spline smoothing
        tck = interpolate.splrep(original_times, values, s=smooth_factor*len(values))
        interpolated = interpolate.splev(target_times, tck)
        
        # Clamp ai valori originali per evitare oscillazioni eccessive
        min_val, max_val = np.min(values), np.max(values)
        interpolated = np.clip(interpolated, min_val, max_val)
        
        return interpolated
        
    except Exception:
        return method_linear_fallback(original_times, values, target_times)

def method_linear_fallback(original_times, values, target_times):
    """
    Fallback lineare sicuro
    """
    if len(original_times) <= 1:
        return np.full(len(target_times), values[0] if len(values) > 0 else np.nan)
    
    interp_func = interpolate.interp1d(original_times, values,
                                     kind='linear', bounds_error=False, 
                                     fill_value='extrapolate')
    return interp_func(target_times)

def method_binning_average(original_times, values, target_times, bin_width_minutes=7.5):
    """
    Metodo binning: divide il tempo in finestre e fa la media
    Utile quando hai più misurazioni vicine nel tempo
    """
    result = np.full(len(target_times), np.nan)
    
    for i, target_time in enumerate(target_times):
        # Definisci finestra temporale centrata sul target
        window_start = target_time - bin_width_minutes
        window_end = target_time + bin_width_minutes
        
        # Trova punti nella finestra
        in_window = (original_times >= window_start) & (original_times <= window_end)
        
        if np.any(in_window):
            # Media pesata per distanza (punti più vicini pesano di più)
            window_times = original_times[in_window]
            window_values = values[in_window]
            
            distances = np.abs(window_times - target_time)
            weights = np.exp(-distances / bin_width_minutes)  # Peso esponenziale
            
            result[i] = np.average(window_values, weights=weights)
    
    # Riempi NaN con interpolazione
    mask = ~np.isnan(result)
    if np.sum(mask) > 1:
        result[~mask] = np.interp(target_times[~mask], target_times[mask], result[mask])
    elif np.sum(mask) == 1:
        result[~mask] = result[mask][0]
    
    return result

def adaptive_alignment(original_times, values, target_times, 
                      regularity_threshold=0.3):
    """
    Metodo adattivo: sceglie automaticamente il miglior metodo di interpolazione
    basandosi sulla regolarità dei dati
    """
    if len(original_times) <= 2:
        return method_linear_fallback(original_times, values, target_times)
    
    # Calcola coefficiente di variazione degli intervalli
    intervals = np.diff(original_times)
    cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf')
    
    print(f"  Coefficiente variazione intervalli: {cv:.3f}")
    
    if cv < regularity_threshold:
        # Dati abbastanza regolari: usa interpolazione lineare
        print("  -> Usando interpolazione lineare")
        return method_linear_fallback(original_times, values, target_times)
    
    elif cv < regularity_threshold * 2:
        # Dati moderatamente irregolari: usa spline con smoothing
        print("  -> Usando spline smoothing")
        return method_spline_smooth(original_times, values, target_times)
    
    else:
        # Dati molto irregolari: usa binning
        print("  -> Usando binning con media pesata")
        return method_binning_average(original_times, values, target_times)

def align_with_quality_metrics(pkl_file_path, output_csv_path, method='adaptive'):
    """
    Versione principale con metriche di qualità dell'allineamento
    """
    with open(pkl_file_path, 'rb') as f:
        metrics_dicts = pickle.load(f)
    
    target_times = np.arange(0, 96 * 15, 15)  # 0, 15, 30, ..., 1425 minuti
    aligned_data = {}
    quality_stats = {'interpolation_errors': 0, 'short_series': 0, 'total_series': 0}
    
    for metric_name, metric_data in metrics_dicts.items():
        print(f"\nProcessando metrica: {metric_name}")
        
        for sample_id, values in metric_data.items():
            quality_stats['total_series'] += 1
            
            if len(values) < 5:  # Serie troppo corte
                quality_stats['short_series'] += 1
                print(f"  ATTENZIONE: {sample_id} ha solo {len(values)} punti")
                aligned_data[sample_id] = np.full(96, np.mean(values) if len(values) > 0 else 0)
                continue
            
            # Crea timestamp
            original_times = np.linspace(0, (len(values)-1) * 15, len(values))
            
            try:
                # Applica metodo selezionato
                if method == 'adaptive':
                    aligned_values = adaptive_alignment(original_times, values, target_times)
                elif method == 'pandas':
                    aligned_values = method_pandas_resample(original_times, values)
                elif method == 'spline':
                    aligned_values = method_spline_smooth(original_times, values, target_times)
                elif method == 'binning':
                    aligned_values = method_binning_average(original_times, values, target_times)
                else:
                    aligned_values = method_linear_fallback(original_times, values, target_times)
                
                aligned_data[sample_id] = aligned_values
                
            except Exception as e:
                quality_stats['interpolation_errors'] += 1
                print(f"  ERRORE con {sample_id}: {e}")
                # Fallback: media costante
                aligned_data[sample_id] = np.full(96, np.mean(values))
    
    # Salva risultati
    time_columns = [f"time_{i}" for i in range(96)]
    rows = [[sample_id] + values.tolist() for sample_id, values in aligned_data.items()]
    df = pd.DataFrame(rows, columns=['dish_well'] + time_columns)
    df.to_csv(output_csv_path, index=False)
    
    # Report qualità
    print(f"\n=== REPORT QUALITÀ ===")
    print(f"Serie totali processate: {quality_stats['total_series']}")
    print(f"Serie con <5 punti: {quality_stats['short_series']}")
    print(f"Errori di interpolazione: {quality_stats['interpolation_errors']}")
    print(f"Successo: {quality_stats['total_series'] - quality_stats['interpolation_errors'] - quality_stats['short_series']}")
    
    return df

# Esempio d'uso con confronto metodi
if __name__ == "__main__":
    pkl_file = "your_metrics.pkl"
    
    # Prova diversi metodi
    methods = ['linear', 'adaptive', 'spline', 'binning']
    
    for method in methods:
        output_file = f"aligned_timeseries_{method}.csv"
        print(f"\n{'='*50}")
        print(f"METODO: {method.upper()}")
        print(f"{'='*50}")
        
        df = align_with_quality_metrics(pkl_file, output_file, method=method)
        
        # Statistiche di base
        numeric_cols = [col for col in df.columns if col.startswith('time_')]
        print(f"Media generale: {df[numeric_cols].mean().mean():.4f}")
        print(f"Std generale: {df[numeric_cols].std().mean():.4f}")
        print(f"Range: {df[numeric_cols].min().min():.4f} - {df[numeric_cols].max().max():.4f}")










"""
# 1. Analisi esplorativa
df = process_with_quality_check("your_metrics.pkl", "output.csv")

# 2. Se hai problemi, prova il metodo adattivo
df_adaptive = align_with_quality_metrics("your_metrics.pkl", "output_adaptive.csv", method='adaptive')

# 3. Confronta visualmente alcuni campioni
import matplotlib.pyplot as plt

sample_cols = [col for col in df.columns if col.startswith('time_')]
plt.figure(figsize=(12, 6))
for i in range(min(3, len(df))):
    plt.plot(df.iloc[i][sample_cols].values, label=f'Sample {i}', alpha=0.7)
plt.xlabel('Time points (15min intervals)')  
plt.ylabel('Metric value')
plt.legend()
plt.show()
"""