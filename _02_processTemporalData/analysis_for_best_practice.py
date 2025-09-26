"""
analyze_time_series_normalization.py

Scopo: analizzare molte serie temporali (map id -> 1D array) e produrre
metriche riassuntive per decidere se normalizzare per-serie o globalmente.

Uso:
    1) Metti qui i tuoi .pkl/.npy nella cartella indicata da DATA_DIR
    2) python analyze_time_series_normalization.py
    3) Guarda outputs: summary_table.csv + qualche immagine .png + output terminale

Dipendenze:
    numpy, pandas, matplotlib
    (scipy è opzionale: usato per skew/kurtosis se disponibile)
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIG ---
DATA_DIR = "/home/phd2/Scrivania/CorsoRepo/cellPIV/datasets/Farneback/pickles"   # metti qui la cartella con i tuoi .pkl/.npy
OUT_DIR = "/home/phd2/Scrivania/CorsoRepo/cellPIV/datasets/Farneback/analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# --- helper: robust skew/kurtosis (usa scipy se possibile) ---
try:
    from scipy.stats import skew, kurtosis
    def _skew(x): return float(skew(x, nan_policy="omit"))
    def _kurt(x): return float(kurtosis(x, fisher=True, nan_policy="omit"))
except Exception:
    def _skew(x):
        x = np.asarray(x)
        x = x[~np.isnan(x)]
        m = x.mean()
        s = x.std(ddof=0) if x.size>0 else np.nan
        return float(np.mean(((x-m)/s)**3)) if s and x.size>0 else np.nan
    def _kurt(x):
        x = np.asarray(x)
        x = x[~np.isnan(x)]
        m = x.mean()
        s = x.std(ddof=0) if x.size>0 else np.nan
        return float(np.mean(((x-m)/s)**4) - 3) if s and x.size>0 else np.nan

# --- load all series from a folder of .pkl/.npy ---
def load_series_from_folder(folder):
    series_dict = {}
    patterns = ["*.pkl", "*.pickle", "*.npy"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(folder, p)))
    for f in files:
        try:
            if f.lower().endswith((".npy",)):
                obj = np.load(f, allow_pickle=True)
            else:
                with open(f, "rb") as fh:
                    obj = pickle.load(fh)
        except Exception as e:
            print(f"WARNING: non ho potuto caricare {f}: {e}")
            continue

        # Se è un dict {id: array}, aggiungilo; se è una singola serie, usa filename
        if isinstance(obj, dict):
            for k, v in obj.items():
                arr = np.asarray(v).astype(float)
                series_dict[k] = arr
        elif isinstance(obj, (list, np.ndarray)):
            # nome derivato dal file
            key = os.path.splitext(os.path.basename(f))[0]
            series_dict[key] = np.asarray(obj).astype(float)
        else:
            # ignora altri tipi
            print(f"INFO: file {f} contiene oggetto di tipo {type(obj)}, ignorato")
    return series_dict

# --- metriche per singola serie ---
def compute_metrics(arr):
    arr = np.asarray(arr).astype(float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n == 0:
        return {}
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
    # autocorrelation lag1 (biased)
    if n > 1:
        a = arr - mean
        denom = (a*a).sum()
        num = (a[:-1]*a[1:]).sum()
        ac1 = float(num/denom) if denom!=0 else np.nan
    else:
        ac1 = np.nan
    return dict(
        n=n, mean=mean, std=std, var=var, median=med, q25=q25, q75=q75,
        p05=p05, p95=p95, range=rng, cv=cv, mad=mad, skew=skewness, kurt=kurt,
        ac1=ac1
    )

# --- main ---
def run_analysis(data_dir, out_dir):
    series = load_series_from_folder(data_dir)
    if len(series)==0:
        print("Nessuna serie caricata. Controlla DATA_DIR.")
        return
    
    # Dictionary Comprehension per troncare tutti i valori a max 300
    do_cut = True
    if do_cut:
        cut_series = {k: v[:300] for k, v in series.items()}

        # Esempio di verifica (stampiamo la lunghezza di alcuni valori)
        print(f"Originale: {len(series['D2018.07.11_S02062_I0141_D_3'])}")
        print(f"Nuovo:     {len(cut_series['D2018.07.11_S02062_I0141_D_3'])}")
        print(f"Originale: {len(series['D2016.10.29_S1633_I141_1'])}")
        print(f"Nuovo:     {len(cut_series['D2016.10.29_S1633_I141_1'])}")

        series = cut_series

    rows = []
    for key, arr in series.items():
        m = compute_metrics(arr)
        if m:
            m["id"] = key
            rows.append(m)
    df = pd.DataFrame(rows).set_index("id")
    # salvataggio
    df.to_csv(os.path.join(out_dir, "summary_table.csv"))

    # --- statistiche across-series ---
    mean_of_means = df["mean"].mean()
    std_of_means = df["mean"].std(ddof=0)
    var_between_means = df["mean"].var(ddof=0)
    mean_within_var = df["var"].mean()
    cv_means = std_of_means / abs(mean_of_means) if mean_of_means!=0 else np.nan

    stds = df["std"].dropna()
    cv_stds = stds.std(ddof=0) / (stds.mean() if stds.mean()!=0 else np.nan)

    # ratio between-series var of means vs average within-series variance
    ratio_between_within = var_between_means / mean_within_var if mean_within_var>0 else np.nan

    # print riepilogo
    print("\n--- RIEPILOGO ---")
    print(f"Numero serie: {len(df)}")
    print(f"Media delle mean (across series): {mean_of_means:.4f}")
    print(f"Std delle mean (across series): {std_of_means:.4f}")
    print(f"Varianza between-means: {var_between_means:.6f}")
    print(f"Avg within-series variance: {mean_within_var:.6f}")
    print(f"Ratio between/within: {ratio_between_within:.4f}")
    print(f"CV of series means: {cv_means:.4f}")
    print(f"CV of series stds: {cv_stds:.4f}")

    # suggerimento di massima (regole motivate sotto)
    suggestion = []
    if (ratio_between_within > 0.5) or (cv_means > 0.2):
        suggestion.append("centratura PER-SERIE consigliata (serie hanno baseline diverse).")
    else:
        suggestion.append("centratura GLOBALE sufficiente (baseline simili).")
    if cv_stds > 0.2:
        suggestion.append("scaling PER-SERIE consigliato (deviazioni standard variano molto).")
    else:
        suggestion.append("scaling GLOBALE sufficiente (std simili tra serie).")

    print("\nSUGGERIMENTO AUTOMATICO:")
    for s in suggestion:
        print(" -", s)

    # --- plots ---
    # 1) histogram of series means
    plt.figure()
    plt.hist(df["mean"].dropna(), bins=40)
    plt.title("Distribuzione delle mean (across series)")
    plt.xlabel("mean")
    plt.ylabel("count")
    plt.savefig(os.path.join(out_dir, "hist_means.png"))
    plt.close()

    # 2) histogram of series stds
    plt.figure()
    plt.hist(df["std"].dropna(), bins=40)
    plt.title("Distribuzione delle std (across series)")
    plt.xlabel("std")
    plt.ylabel("count")
    plt.savefig(os.path.join(out_dir, "hist_stds.png"))
    plt.close()

    # 3) scatter mean vs std
    plt.figure()
    plt.scatter(df["mean"], df["std"], s=10)
    plt.title("mean vs std (per serie)")
    plt.xlabel("mean")
    plt.ylabel("std")
    plt.savefig(os.path.join(out_dir, "scatter_mean_std.png"))
    plt.close()

    # 4) boxplot lunghezze
    plt.figure()
    plt.boxplot(df["n"].dropna().values, vert=True)
    plt.title("Boxplot lunghezze serie")
    plt.ylabel("n_frames")
    plt.savefig(os.path.join(out_dir, "boxplot_lengths.png"))
    plt.close()

    # 5) optional: show numeric summary
    with open(os.path.join(out_dir, "analysis_report.txt"), "w") as fh:
        fh.write("RIEPILOGO\n")
        fh.write(f"Numero serie: {len(df)}\n")
        fh.write(f"mean_of_means: {mean_of_means}\n")
        fh.write(f"std_of_means: {std_of_means}\n")
        fh.write(f"var_between_means: {var_between_means}\n")
        fh.write(f"mean_within_var: {mean_within_var}\n")
        fh.write(f"ratio_between_within: {ratio_between_within}\n")
        fh.write(f"cv_means: {cv_means}\n")
        fh.write(f"cv_stds: {cv_stds}\n\n")
        fh.write("Suggerimento:\n")
        for s in suggestion:
            fh.write(" - " + s + "\n")

    print(f"\nOutputs salvati in: {out_dir}")
    print("File: summary_table.csv, analysis_report.txt, hist_means.png, hist_stds.png, scatter_mean_std.png, boxplot_lengths.png")
    return df, suggestion

if __name__ == "__main__":
    df, suggestion = run_analysis(DATA_DIR, OUT_DIR)
