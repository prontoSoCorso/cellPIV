import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys
# Rileva il percorso della cartella genitore, che sarà la stessa in cui ho il file da convertire
current_dir = os.path.dirname(os.path.abspath(__file__))

# Individua la cartella 'cellPIV' come riferimento
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while os.path.basename(parent_dir) != "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from _utils_._utils import detect_time_columns

# Split dei dati in base a patient_id
def stratified_split(data, train_size, seed):
    # Crea un DataFrame aggregato a livello di `patient_id`
    patient_info = data.groupby('patient_id').agg({
        'BLASTO NY': 'max'  # Etichetta prevalente per ogni paziente
    }).reset_index()

    # Split stratificato a livello di paziente
    train_patients, tmp_patients = train_test_split(
        patient_info,
        train_size=train_size,
        stratify=patient_info['BLASTO NY'],
        random_state=seed
    )
    val_patients, test_patients = train_test_split(
        tmp_patients,
        test_size=0.5,
        stratify=tmp_patients['BLASTO NY'],
        random_state=seed
    )

    # Filtra i dati originali per ottenere i subset
    train_data = data[data['patient_id'].isin(train_patients['patient_id'])]
    val_data = data[data['patient_id'].isin(val_patients['patient_id'])]
    test_data = data[data['patient_id'].isin(test_patients['patient_id'])]

    return train_data, val_data, test_data


def _normalize_array_with_percentiles(a, p_low, p_high, clip=True):
    a = np.asarray(a, dtype=float)
    if a.size == 0 or np.all(np.isnan(a)):
        return a.copy(), np.nan, np.nan
    p1 = float(np.nanpercentile(a, p_low))
    p99 = float(np.nanpercentile(a, p_high))
    denom = p99 - p1
    # fallback: evita divisione per zero, produce serie tutta a 0.0 (più sicuro)
    if denom == 0 or np.isnan(denom):
        norm = np.zeros_like(a, dtype=float)
    else:
        norm = (a - p1) / denom
    if clip:
        norm = np.clip(norm, 0.0, 1.0)
    return norm, p1, p99


def normalize_per_series(train_df, val_df, test_df,
                         time_cols=None,
                         p_low=1.0, p_high=99.0,
                         fallback_low=5.0, fallback_high=95.0,
                         min_points_for_1_99=50,
                         clip=True,
                         ):
    """
    Normalizza per-serie tre DataFrame (train/val/test).

    Input:
      train_df, val_df, test_df: pandas.DataFrame con colonne meta + colonne temporali
      time_cols: lista di colonne temporali; se None, vengono rilevate automaticamente
      p_low,p_high: percentili principali (es. 1,99)
      fallback_low,fallback_high: percentili fallback (es. 5,95)
      min_points_for_1_99: soglia per scegliere p_low/p_high vs fallback
      clip: se True applica clip a [0,1]
      apply_train_stats_to_others: se True, per gli id presenti in train usa i p1/p99 calcolati sul train per val/test

    Output:
      train_norm_df, val_norm_df, test_norm_df, params_dict
      - params_dict[dish_well] = {"p_low_used":..., "p_high_used":..., "p1":..., "p99":...}
    """

    # copia (per non modificare originali)
    train = train_df.copy() if train_df is not None else pd.DataFrame()
    val = val_df.copy() if val_df is not None else pd.DataFrame()
    test = test_df.copy() if test_df is not None else pd.DataFrame()

    # detect time columns if not provided
    if time_cols is None:
        # try from train, else val, else test
        for df in (train, val, test):
            if df is not None and not df.empty:
                time_cols = detect_time_columns(df)
                if len(time_cols) > 0:
                    break
        if time_cols is None:
            time_cols = []

    params = {}

    def choose_percentiles(arr):
        n_valid = int(np.sum(~np.isnan(arr)))
        if n_valid >= min_points_for_1_99:
            return p_low, p_high
        else:
            return fallback_low, fallback_high

    def normalize_df_rows(df):
        out = df.copy()
        for idx, row in df.iterrows():
            arr = row[time_cols].values.astype(float)
            qlow, qhigh = choose_percentiles(arr)
            norm, p1, p99 = _normalize_array_with_percentiles(arr, qlow, qhigh, clip=clip)
            out.loc[idx, time_cols] = norm
            # salva params usando dish_well se presente, altrimenti l'indice
            key = row['dish_well'] if 'dish_well' in df.columns else idx
            params[str(key)] = {"p_low_used": qlow, "p_high_used": qhigh, "p1": p1, "p99": p99}
        return out

    # normalizza tutti i subsets
    print("Normalizing train...")
    train_norm = normalize_df_rows(train)
    print("Normalizing validation...")
    val_norm = normalize_df_rows(val)
    print("Normalizing test...")
    test_norm = normalize_df_rows(test)

    # Replace NaN or infinite values in temporal columns with 0
    for df in (train_norm, val_norm, test_norm):
        df[time_cols] = df[time_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    return train_norm, val_norm, test_norm, params


# Normalizzazione del train con quantile normalization
def normalize_data(train_data, val_data, test_data, inf_quantile=0.10, sup_quantile=0.90):
    # Seleziona solo le colonne temporali da normalizzare
    temporal_columns = detect_time_columns(train_data)

    # Separo i dati temporali
    train_temporal = train_data[temporal_columns]
    val_temporal = val_data[temporal_columns]
    test_temporal = test_data[temporal_columns]

    # Calcolo il 10° e il 90° percentile per il train
    min_val = train_temporal[train_temporal > 0].quantile(inf_quantile).min()
    max_val = train_temporal[train_temporal > 0].quantile(sup_quantile).max()

    print("=====VALORI DI MIN E MAX PER NORMALIZZAZIONE=====")
    print(f"min_val = {min_val}, max_val = {max_val}")

    # Normalizzo il train
    train_normalized = (train_temporal - min_val) / (max_val - min_val)
    train_normalized[train_temporal == 0] = 0

    # Normalizzo il validation usando min e max del train
    val_normalized = (val_temporal - min_val) / (max_val - min_val)
    val_normalized[val_temporal == 0] = 0

    # Normalizzo il test usando min e max del train
    test_normalized = (test_temporal - min_val) / (max_val - min_val)
    test_normalized[test_temporal == 0] = 0

    # Creo nuove copie senza modificare gli originali
    train_data_norm = train_data.copy()
    val_data_norm = val_data.copy()
    test_data_norm = test_data.copy()

    train_data_norm[temporal_columns] = train_normalized
    val_data_norm[temporal_columns] = val_normalized
    test_data_norm[temporal_columns] = test_normalized

    # Replace NaN or infinite values in temporal columns with 0
    for df in (train_data_norm, val_data_norm, test_data_norm):
        df[temporal_columns] = df[temporal_columns].replace([np.inf, -np.inf], np.nan).fillna(0)

    return train_data_norm, val_data_norm, test_data_norm


# Salvataggio dei file normalizzati
def save_data(train_data, val_data, test_data, output_base_path, days_to_consider):
    train_data.to_csv(f"{output_base_path}_train.csv", index=False)
    val_data.to_csv(f"{output_base_path}_val.csv", index=False)
    test_data.to_csv(f"{output_base_path}_test.csv", index=False)

    print(f"Dati salvati con successo per {days_to_consider} giorni.")
    print(f"Train: {output_base_path}_train.csv")
    print(f"Validation: {output_base_path}_val.csv")
    print(f"Test: {output_base_path}_test.csv")
