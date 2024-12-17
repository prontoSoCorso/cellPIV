import pandas as pd
import os

# Ottieni il percorso del file corrente
current_file_path = os.path.abspath(__file__)
# Risali la gerarchia fino alla cartella "cellPIV"
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
import sys
sys.path.append(parent_dir)

from config import Config_00_preprocessing as conf

# Leggi il file Excel senza password (tolta manualmente)
xls = pd.ExcelFile(conf.path_original_excel, engine='openpyxl')

# Leggi i due fogli in DataFrame di pandas
sheet1 = pd.read_excel(xls, sheet_name="blasto NO SI")
sheet2 = pd.read_excel(xls, sheet_name="blasto NON in foglio 1 ")

# Rinomina le colonne del foglio 2 per corrispondere a quelle del foglio 1
column_mapping = {
    "codice escope": "slide",
    "escope well": "well",
    "escope cose_well": "slide_well",
    "PGD_date": "data",
    "tSB": "tSB",
    "tB": "tB",
    "tEB": "tEB",
    "maternal age": "maternal age",
    "sperm quality": "sperm quality",
    "mezzo di coltura": "mezzo di coltura",
    "presente in foglio 1": "BLASTO NY"
}

sheet2.rename(columns=column_mapping, inplace=True)

# Seleziona le righe dal foglio 2 che non sono presenti nel foglio 1
sheet2_filtered = sheet2[sheet2["BLASTO NY"] == 0]

# Imposta "BLASTO NY" con valore 1 per tutte le righe del foglio 2 filtrato
sheet2_filtered["BLASTO NY"] = 1

# Aggiungi le colonne mancanti nel foglio 2 con valori NaN
missing_columns = [col for col in sheet1.columns if col not in sheet2_filtered.columns]
for col in missing_columns:
    sheet2_filtered[col] = pd.NA

# Riordina le colonne del foglio 2 per corrispondere a quelle del foglio 1 (dovrebbero essere già in ordine, ma per essere sicuri)
sheet2_filtered = sheet2_filtered[sheet1.columns]

# Combina i due DataFrame
blasto_labels_df = pd.concat([sheet1, sheet2_filtered], ignore_index=True)

# Rinomina le colonne "slide" a "dish" e "slide_well" a "dish_well"
blasto_labels_df.rename(columns={"slide": "dish", "slide_well": "dish_well"}, inplace=True)

# Save the result to a CSV file
blasto_labels_df.to_csv(conf.path_single_csv, index=False)

print(blasto_labels_df.head())
print("========================")

# Leggi il file "pz con doppia dish" che contiene la mappatura dei pazienti con doppia dish
double_dish_df = pd.read_excel(conf.path_double_dish_excel, engine='openpyxl', header=None, names=['dish1', 'dish2','id'])
double_dish_df = double_dish_df.iloc[:, :2]
print(double_dish_df.head())
print("========================")

# Crea un dizionario per mappare le slide ai pazienti
patient_id = 1
dish_to_patient = {}

# Associa le slide doppie allo stesso paziente
for _, row in double_dish_df.iterrows():
    dish1, dish2 = row['dish1'], row['dish2']
    dish_to_patient[dish1] = patient_id
    dish_to_patient[dish2] = patient_id
    patient_id += 1

# Associa le slide singole ai pazienti rimanenti
for dish in blasto_labels_df['dish'].unique():
    if dish not in dish_to_patient:
        dish_to_patient[dish] = patient_id
        patient_id += 1

# Aggiungi la colonna dell'identificativo del paziente
blasto_labels_df['patient_id'] = blasto_labels_df['dish'].map(dish_to_patient)

# Sposta la colonna "patient_id" come prima colonna
cols = ['patient_id'] + [col for col in blasto_labels_df.columns if col != 'patient_id']
blasto_labels_df = blasto_labels_df[cols]

# Save the result to a CSV file
blasto_labels_df.to_csv(conf.path_addedID_csv, index=False)

print(blasto_labels_df.head())
