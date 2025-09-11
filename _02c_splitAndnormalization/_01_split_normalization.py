import pandas as pd
from sklearn.model_selection import train_test_split


# Caricamento del file CSV
def load_data(csv_file_path, initial_frames_to_cut, max_frames):
    # Load the CSV file
    data = pd.read_csv(csv_file_path)

    # Identify all columns that start with "value_"
    value_columns = [col for col in data.columns if col.startswith("value_")]

    # Selects frames starting at index initial_frames_to_cut for the next max_frames columns.
    pruned_value_columns = value_columns[initial_frames_to_cut: initial_frames_to_cut + max_frames]

    # Identify the meta columns by excluding the ones starting with "value_"
    meta_columns = [col for col in data.columns if not col.startswith("value_")]
    
    # Concatenate meta_columns with the pruned subset of value columns
    pruned_data = data[meta_columns + pruned_value_columns]
    
    return pruned_data


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


# Normalizzazione del train con quantile normalization
def normalize_data(train_data, val_data, test_data, inf_quantile=0.10, sup_quantile=0.90):
    # Seleziona solo le colonne temporali da normalizzare
    temporal_columns = [col for col in train_data.columns if col.startswith("value_")]

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
