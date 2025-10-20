#!/usr/bin/env python3
"""
compare_normalizations.py

Script completo per confrontare diverse strategie di normalizzazione sulle tue
serie temporali (raw, per-serie z-score, globale z-score, center-per-series +
scale-globale) e produrre metriche riassuntive e grafici.

Outputs:
 - summary_raw.csv, summary_per_series.csv, summary_global.csv,
   summary_center_per_series_scale_global.csv
 - analysis_report_comparison.txt
 - grafici (histogram, scatter) nella cartella OUT_DIR

Configurazione: modifica DATA_DIR e OUT_DIR sotto.

Esegue (se presente) anche un controllo di correlazione tra mean/std raw e
labels se fornisci un file CSV con colonne ['id','label'] (label binaria o
numerica).

Dipendenze: numpy, pandas, matplotlib. scipy opzionale (usato per skew/kurt,
pointbiserialr). Lo script è robusto se scipy non è installato.
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
DATA_DIR = "/home/phd2/Scrivania/CorsoRepo/cellPIV/datasets/Farneback/pickles"
OUT_DIR = "/home/phd2/Scrivania/CorsoRepo/cellPIV/datasets/Farneback/analysis_norm_compare"
LABELS_CSV = ""  # opzionale: percorso a CSV con colonne `id,label` (lascia vuoto per disabilitare)
TRUNCATE_MAX = 300  # se vuoi troncare tutte le serie a questa lunghezza (None per no)
EPS = 1e-8
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- helpers ----------------
try:
    from scipy.stats import skew, kurtosis, pointbiserialr
    def _skew(x): return float(skew(x, nan_policy="omit"))
    def _kurt(x): return float(kurtosis(x, fisher=True, nan_policy="omit"))
    _has_scipy = True
except Exception:
    _has_scipy = False
    def _skew(x):
        x = np.asarray(x)
        x = x[~np.isnan(x)]
        if x.size == 0:
            return np.nan
        m = x.mean()
        s = x.std(ddof=0)
        if s == 0:
            return np.nan
        return float(np.mean(((x-m)/s)**3))
    def _kurt(x):
        x = np.asarray(x)
        x = x[~np.isnan(x)]
        if x.size == 0:
            return np.nan
        m = x.mean()
        s = x.std(ddof=0)
        if s == 0:
            return np.nan
        return float(np.mean(((x-m)/s)**4) - 3)

# Caricamento delle serie (usa lo stesso comportamento del tuo script precedente)

def load_series_from_folder(folder, patterns=("*.pkl","*.pickle","*.npy")):
    series_dict = {}
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(folder, p)))
    for f in sorted(files):
        try:
            if f.lower().endswith('.npy'):
                obj = np.load(f, allow_pickle=True)
            else:
                with open(f, 'rb') as fh:
                    obj = pickle.load(fh)
        except Exception as e:
            print(f"WARNING: non ho potuto caricare {f}: {e}")
            continue

        if isinstance(obj, dict):
            for k, v in obj.items():
                try:
                    arr = np.asarray(v).astype(float)
                except Exception:
                    arr = np.asarray(v, dtype=float)
                series_dict[k] = arr
        elif isinstance(obj, (list, np.ndarray)):
            key = os.path.splitext(os.path.basename(f))[0]
            series_dict[key] = np.asarray(obj).astype(float)
        else:
            print(f"INFO: file {f} contiene oggetto di tipo {type(obj)}, ignorato")
    return series_dict

# Metriche per singola serie (incluso skew/kurt)

def compute_metrics(arr):
    arr = np.asarray(arr).astype(float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n == 0:
        return None
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    var = float(std*std)
    med = float(np.median(arr))
    q25, q75 = float(np.percentile(arr,25)), float(np.percentile(arr,75))
    p05, p95 = float(np.percentile(arr,5)), float(np.percentile(arr,95))
    rng = float(arr.max() - arr.min()) if n>0 else np.nan
    cv = float(std/mean) if mean!=0 else np.nan
    mad = float(np.median(np.abs(arr - med)))
    skewness = _skew(arr)
    kurt = _kurt(arr)
    if n > 1:
        a = arr - mean
        denom = (a*a).sum()
        num = (a[:-1]*a[1:]).sum()
        ac1 = float(num/denom) if denom!=0 else np.nan
    else:
        ac1 = np.nan
    return dict(n=n, mean=mean, std=std, var=var, median=med, q25=q25, q75=q75,
                p05=p05, p95=p95, range=rng, cv=cv, mad=mad, skew=skewness, kurt=kurt, ac1=ac1)

# Funzioni di normalizzazione richieste

def normalize_per_series(series_dict, eps=EPS):
    out = {}
    for k, arr in series_dict.items():
        a = np.asarray(arr).astype(float)
        mean = np.nanmean(a)
        a = a - mean
        s = np.nanstd(a)
        if s < eps or np.isnan(s):
            s = 1.0
        out[k] = (a / s)
    return out


def normalize_global(series_dict, eps=EPS):
    all_vals = np.concatenate([np.ravel(v) for v in series_dict.values() if np.asarray(v).size>0])
    global_mean = np.nanmean(all_vals)
    global_std = np.nanstd(all_vals)
    if global_std < eps or np.isnan(global_std):
        global_std = 1.0
    out = {k: (np.asarray(v).astype(float) - global_mean) / global_std for k, v in series_dict.items()}
    return out, global_mean, global_std


def center_per_series_scale_global(series_dict, eps=EPS):
    # sottraggo mean per serie, ma divido per std globale
    all_vals = np.concatenate([np.ravel(v) for v in series_dict.values() if np.asarray(v).size>0])
    global_std = np.nanstd(all_vals)
    if global_std < eps or np.isnan(global_std):
        global_std = 1.0
    out = {}
    for k, v in series_dict.items():
        a = np.asarray(v).astype(float)
        a = a - np.nanmean(a)
        out[k] = a / global_std
    return out, global_std

# Calcolo di metriche di riepilogo per ogni dizionario di serie

def compute_summary_metrics(series_dict):
    rows = []
    for k, v in series_dict.items():
        a = np.asarray(v).astype(float)
        a = a[~np.isnan(a)]
        if a.size == 0:
            continue
        rows.append({"id": k,
                     "n": int(a.size),
                     "mean": float(a.mean()),
                     "std": float(a.std(ddof=0)),
                     "var": float(a.var(ddof=0))})
    df = pd.DataFrame(rows).set_index('id').sort_index()
    return df


def between_within_ratio(df_summary):
    mean_of_means = df_summary['mean'].mean()
    var_between_means = df_summary['mean'].var(ddof=0)
    mean_within_var = df_summary['var'].mean()
    ratio = var_between_means / mean_within_var if mean_within_var>0 else np.nan
    cv_means = (df_summary['mean'].std(ddof=0)/abs(mean_of_means)) if mean_of_means!=0 else np.nan
    cv_stds = (df_summary['std'].std(ddof=0)/df_summary['std'].mean()) if df_summary['std'].mean()!=0 else np.nan
    return dict(mean_of_means=mean_of_means, var_between_means=var_between_means,
                mean_within_var=mean_within_var, ratio=ratio, cv_means=cv_means, cv_stds=cv_stds)

# Funzione per salvare DataFrame e stampare breve riepilogo

def save_summary(df, path):
    df.to_csv(path)

# Funzione per provare correlazioni mean/std raw con labels (opzionale)

def analyze_label_correlations(raw_df, labels_csv_path):
    if not labels_csv_path:
        print("Labels non fornite: salto analisi correlazioni.")
        return None
    if not os.path.exists(labels_csv_path):
        print(f"File labels non trovato: {labels_csv_path}")
        return None
    labs = pd.read_csv(labels_csv_path)
    if 'id' not in labs.columns or 'label' not in labs.columns:
        print("Il CSV delle labels deve contenere colonne 'id' e 'label'.")
        return None
    merged = raw_df.join(labs.set_index('id'), how='inner')
    if merged.shape[0] == 0:
        print("Nessuna corrispondenza tra ids delle serie e labels.")
        return None

    out = {}
    labels = merged['label'].values
    means = merged['mean'].values
    stds = merged['std'].values

    # se label binaria (0/1), preferiamo pointbiserialr
    unique = np.unique(labels)
    try:
        if _has_scipy and (set(unique) <= {0,1} or unique.size==2):
            r_mean, p_mean = pointbiserialr(labels, means)
            r_std, p_std = pointbiserialr(labels, stds)
            out['pointbiserial_mean'] = (r_mean, p_mean)
            out['pointbiserial_std'] = (r_std, p_std)
        else:
            # fallback: pearson between continuous label and mean/std
            from scipy.stats import pearsonr
            r_mean, p_mean = pearsonr(labels, means)
            r_std, p_std = pearsonr(labels, stds)
            out['pearson_mean'] = (r_mean, p_mean)
            out['pearson_std'] = (r_std, p_std)
    except Exception as e:
        print(f"Impossibile calcolare correlazioni tramite scipy: {e}. Calcolo statistiche di base.")
        # calcolo differenze di media per label binaria
        if unique.size==2:
            lab0 = means[labels==unique[0]]
            lab1 = means[labels==unique[1]]
            out['mean_diff_mean'] = float(np.nanmean(lab1)-np.nanmean(lab0))
            lab0s = stds[labels==unique[0]]
            lab1s = stds[labels==unique[1]]
            out['mean_diff_std'] = float(np.nanmean(lab1s)-np.nanmean(lab0s))
    return out

# ---------------- main flow ----------------

def main():
    print("Caricamento serie da:", DATA_DIR)
    series = load_series_from_folder(DATA_DIR)
    if len(series)==0:
        print("Nessuna serie caricata. Controlla DATA_DIR.")
        return

    # opzionale: tronca
    if TRUNCATE_MAX is not None:
        series = {k: v[:TRUNCATE_MAX] for k, v in series.items()}

    # raw summary
    raw_df = compute_summary_metrics(series)
    save_summary(raw_df, os.path.join(OUT_DIR, 'summary_raw.csv'))
    raw_stats = between_within_ratio(raw_df)
    print("RAW stats:", raw_stats)

    # per-series normalization
    per_series = normalize_per_series(series)
    per_df = compute_summary_metrics(per_series)
    save_summary(per_df, os.path.join(OUT_DIR, 'summary_per_series.csv'))
    per_stats = between_within_ratio(per_df)
    print("PER-SERIE stats:", per_stats)

    # global normalization
    global_norm, gmean, gstd = normalize_global(series)
    glob_df = compute_summary_metrics(global_norm)
    save_summary(glob_df, os.path.join(OUT_DIR, 'summary_global.csv'))
    glob_stats = between_within_ratio(glob_df)
    print("GLOBALE stats:", glob_stats)

    # center-per-series + scale-global
    cpsg, global_std_used = center_per_series_scale_global(series)
    cpsg_df = compute_summary_metrics(cpsg)
    save_summary(cpsg_df, os.path.join(OUT_DIR, 'summary_center_per_series_scale_global.csv'))
    cpsg_stats = between_within_ratio(cpsg_df)
    print("CENTER-PER-SERIE + SCALE-GLOBALE stats:", cpsg_stats)

    # Salva report testuale
    report_lines = []
    report_lines.append("COMPARISON REPORT\n")
    report_lines.append(f"N_series: {len(raw_df)}\n")
    report_lines.append("--- RAW ---\n")
    for k,v in raw_stats.items():
        report_lines.append(f"{k}: {v}\n")
    report_lines.append("--- PER-SERIE (z per serie) ---\n")
    for k,v in per_stats.items():
        report_lines.append(f"{k}: {v}\n")
    report_lines.append("--- GLOBALE (z con mean/std globali) ---\n")
    report_lines.append(f"global_mean: {gmean}, global_std: {gstd}\n")
    for k,v in glob_stats.items():
        report_lines.append(f"{k}: {v}\n")
    report_lines.append("--- CENTER-PER-SERIE + SCALE-GLOBALE ---\n")
    for k,v in cpsg_stats.items():
        report_lines.append(f"{k}: {v}\n")

    # Analisi correlazioni (opzionale)
    corr_res = analyze_label_correlations(raw_df, LABELS_CSV) if LABELS_CSV else None
    report_lines.append("\nLabel correlation analysis:\n")
    report_lines.append(str(corr_res) + "\n")

    with open(os.path.join(OUT_DIR, 'analysis_report_comparison.txt'), 'w') as fh:
        fh.writelines(report_lines)

    # ---------------- plots ----------------
    # histogram delle mean raw vs per_series vs global
    plt.figure(figsize=(10,6))
    plt.subplot(2,2,1)
    plt.hist(raw_df['mean'].dropna(), bins=40)
    plt.title('raw means')
    plt.subplot(2,2,2)
    plt.hist(per_df['mean'].dropna(), bins=40)
    plt.title('per-series means (should be ~0)')
    plt.subplot(2,2,3)
    plt.hist(glob_df['mean'].dropna(), bins=40)
    plt.title('global-normalized means')
    plt.subplot(2,2,4)
    plt.hist(cpsg_df['mean'].dropna(), bins=40)
    plt.title('center-per-series + scale-global means')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hist_means_comparison.png'))
    plt.close()

    # scatter mean vs std for raw and per-series
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.scatter(raw_df['mean'], raw_df['std'], s=8)
    plt.title('raw: mean vs std')
    plt.xlabel('mean'); plt.ylabel('std')
    plt.subplot(1,2,2)
    plt.scatter(per_df['mean'], per_df['std'], s=8)
    plt.title('per-series normalized: mean vs std')
    plt.xlabel('mean'); plt.ylabel('std')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'scatter_mean_std_raw_vs_per.png'))
    plt.close()

    # boxplot lunghezze
    plt.figure()
    plt.boxplot(raw_df['n'].dropna().values, vert=True)
    plt.title('Boxplot lunghezze serie (frames)')
    plt.ylabel('n_frames')
    plt.savefig(os.path.join(OUT_DIR, 'boxplot_lengths.png'))
    plt.close()

    # salva tutti i summary csv
    raw_df.to_csv(os.path.join(OUT_DIR, 'summary_raw.csv'))
    per_df.to_csv(os.path.join(OUT_DIR, 'summary_per_series.csv'))
    glob_df.to_csv(os.path.join(OUT_DIR, 'summary_global.csv'))
    cpsg_df.to_csv(os.path.join(OUT_DIR, 'summary_center_per_series_scale_global.csv'))

    print('\nOutputs salvati in:', OUT_DIR)
    print('Files principali: summary_raw.csv, summary_per_series.csv, summary_global.csv, summary_center_per_series_scale_global.csv, analysis_report_comparison.txt')

if __name__ == '__main__':
    main()
