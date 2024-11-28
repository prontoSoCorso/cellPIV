import pandas as pd
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

from config import Config_02_temporalData as conf

# Percorsi dei file
temporal_data_path = conf.temporal_csv_path
labels_data_path = conf.csv_file_Danilo_path
output_path = conf.output_final_csv_path

# Leggi i file CSV
temporal_data = pd.read_csv(temporal_data_path)
labels_data = pd.read_csv(labels_data_path)

'''
Purtroppo ci sono duplicati nell'excel, trovati con:

# Controlla duplicati in temporal_data
duplicates_temporal = temporal_data[temporal_data.duplicated(subset=["dish_well"], keep=False)]
print(f"Duplicati in temporal_data:\n{duplicates_temporal}")

# Controlla duplicati in labels_data
duplicates_labels = labels_data[labels_data.duplicated(subset=["dish_well"], keep=False)]
print(f"Duplicati in labels_data:\n{duplicates_labels}")

Li elimino
'''

# Elimino duplicati
temporal_data = temporal_data.drop_duplicates(subset=["dish_well"])
labels_data = labels_data.drop_duplicates(subset=["dish_well"])

# Mantieni solo le colonne necessarie
columns_to_keep = ["dish_well"] + [col for col in temporal_data.columns if col.startswith("time_")]
temporal_data = temporal_data[columns_to_keep]

# Unisci i due file usando la colonna "dish_well"
merged_data = pd.merge(labels_data, temporal_data, on="dish_well", how="inner")

# Seleziona solo le colonne richieste
columns_to_keep_final = ["patient_id", "dish_well", "BLASTO NY"] + [col for col in temporal_data.columns if col.startswith("time_")]
merged_data = merged_data[columns_to_keep_final]

# Rinomina le colonne temporali in value_1, value_2, ..., value_672
new_column_names = ["patient_id", "dish_well", "BLASTO NY"] + [f"value_{i+1}" for i in range(len(columns_to_keep_final) - 3)]
merged_data.columns = new_column_names

# Salva il nuovo file CSV
merged_data.to_csv(output_path, index=False)

print(f"File CSV unito salvato in: {output_path}")

''' 
    datasets = pd.read_csv(output_path)
    nunq = datasets.iloc[:,0].nunique()
    print(nunq)
    dim = datasets.shape
    print(dim)

    597 pazienti!!! 
    2462 serie temporali 
    672 frame           '''
