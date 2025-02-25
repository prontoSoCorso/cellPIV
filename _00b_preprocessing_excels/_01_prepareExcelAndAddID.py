import pandas as pd

def merge_sheets_and_prepare_csv(path_original_excel):
    # Leggi il file Excel (nuovo formato)
    new_excel = pd.ExcelFile(path_original_excel, engine='openpyxl')

    # Leggi il foglio "lista"
    blasto_labels_df = pd.read_excel(new_excel, sheet_name="lista")

    # Rinomina colonne come richiesto (se necessario)
    column_mapping = {
        "slide": "dish",
        "slide_well": "dish_well",
        "blasto ny" : "BLASTO NY"
    }
    blasto_labels_df.rename(columns=column_mapping, inplace=True)

    return blasto_labels_df



def match_double_dishes_and_create_csv(path_double_dish_excel, blasto_labels_df, output_path_added_ID):
    # Leggi il file "pz con doppia dish" per creare la mappatura
    double_dish_df = pd.read_excel(path_double_dish_excel, engine='openpyxl', header=None, names=['dish1', 'dish2', 'id'])
    double_dish_df = double_dish_df.iloc[:, :2]

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

    # Salva il risultato in un file CSV
    blasto_labels_df.to_csv(output_path_added_ID, index=False)

    print("\n===== Elaborazione completata. File salvato in:", output_path_added_ID, "=====\n")
