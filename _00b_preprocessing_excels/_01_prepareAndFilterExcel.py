import pandas as pd
import os
import sys
import logging
from pathlib import Path
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# Configurazione dei percorsi e dei parametri
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path) 
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_00_preprocessing as conf

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


def filter_by_valid_wells(final_table_path: str,
                          valid_csv_path: str,
                          output_path: str):
    """
    final_table_path : str
        Path to your merged CSV/Excel with a 'dish_well' column.
    valid_csv_path : str
        Path to valid_wells_acquisition_times.csv (dish_well, acquisition_hours).
    output_path : str
        Where to write the filtered table (same format as input).
    """
    # Check if input paths exist
    if not os.path.exists(final_table_path):
        raise FileNotFoundError(f"Final table path does not exist: {final_table_path}")
    if not os.path.exists(valid_csv_path):
        raise FileNotFoundError(f"Valid wells CSV path does not exist: {valid_csv_path}")
    # Check if output path is valid
    if not os.path.isdir(os.path.dirname(output_path)):
        raise NotADirectoryError(f"Output path directory does not exist: {os.path.dirname(output_path)}")
    
    # load valid dish_well keys
    valid_df = pd.read_csv(valid_csv_path, usecols=["dish_well"])
    valid_set = set(valid_df["dish_well"].astype(str))

    # load final table (CSV or XLSX)
    ext = os.path.splitext(final_table_path)[1].lower()
    if ext in (".xls", ".xlsx"):
        df = pd.read_excel(final_table_path, engine="openpyxl")
    else:
        df = pd.read_csv(final_table_path)

    if "dish_well" not in df.columns:
        raise ValueError(f"'dish_well' column not found in {final_table_path}")

    # filter
    filtered = df[df["dish_well"].astype(str).isin(valid_set)].copy()

    # write out
    out_ext = os.path.splitext(output_path)[1].lower()
    if out_ext in (".xls", ".xlsx"):
        filtered.to_excel(output_path, index=False)
    else:
        filtered.to_csv(output_path, index=False)

    print(f"Filtered table saved to {output_path}, {len(filtered)} rows (of {len(df)})")


if __name__ == "__main__":
    """
    Main semplice e inline per:
      1) merge_sheets_and_prepare_csv
      2) match_double_dishes_and_create_csv (salva CSV con patient_id)
      3) opzionale: filter_by_valid_wells

    Configurazione: modifica le variabili qui sotto se vuoi cambiare comportamento/percorsi.
    """
    import logging
    from pathlib import Path
    import time

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    start_time = time.time()

    # -----------------------
    # Opzioni (modifica qui)
    # -----------------------
    SAVE_OUTPUT = True        # se False, salta il salvataggio del CSV con patient_id
    APPLY_FILTER = True       # se True tenta di applicare filter_by_valid_wells (richiede valid_wells_path)
    # -----------------------

    # directory di base (cartella dello script)
    base_dir = Path(__file__).resolve().parent

    # Prova a prendere i percorsi da conf se definito, altrimenti usa valori di default nella cartella dello script
    try:
        _conf = conf  # conf dovrebbe essere importato in testa allo script
    except NameError:
        _conf = None

    path_input_excel = getattr(_conf, "path_original_excel", None) or str(base_dir / "original.xlsx")
    path_double_dish = getattr(_conf, "path_double_dish_excel", None) or str(base_dir / "double_dish.xlsx")
    path_added_id = getattr(_conf, "path_addedID_csv", None) or str(base_dir / "added_id.csv")
    # percorso CSV dei valid wells (se non presente, verrà cercato in base_dir)
    path_valid_wells = getattr(_conf, "valid_wells_file", None) or str(base_dir / "datasets/valid_wells_acquisition_times.csv")
    filtered_blasto_dataset = getattr(_conf, "filtered_blasto_dataset", None) or str(base_dir / "filtered_table.csv")

    # Converti in Path per comodità
    path_input_excel = Path(path_input_excel)
    path_double_dish = Path(path_double_dish)
    path_added_id = Path(path_added_id)
    path_valid_wells = Path(path_valid_wells)
    filtered_blasto_dataset = Path(filtered_blasto_dataset)

    try:
        logging.info("Percorsi usati:")
        logging.info("  input excel: %s", path_input_excel)
        logging.info("  double dish: %s", path_double_dish)
        logging.info("  output added_id: %s", path_added_id)
        logging.info("  valid_wells (atteso): %s", path_valid_wells)
        logging.info("  filtered output: %s", filtered_blasto_dataset)
        logging.info("Opzioni: SAVE_OUTPUT=%s, APPLY_FILTER=%s", SAVE_OUTPUT, APPLY_FILTER)

        # controlli preliminari
        if not path_input_excel.exists():
            logging.error("File input excel non trovato: %s", path_input_excel)
            raise FileNotFoundError(f"Input excel non trovato: {path_input_excel}")
        if not path_double_dish.exists():
            logging.error("File double-dish non trovato: %s", path_double_dish)
            raise FileNotFoundError(f"Double-dish file non trovato: {path_double_dish}")

        # assicurati che le cartelle di destinazione esistano
        path_added_id.parent.mkdir(parents=True, exist_ok=True)
        filtered_blasto_dataset.parent.mkdir(parents=True, exist_ok=True)

        # 1) merge sheets
        logging.info("Step 1/3 — merge_sheets_and_prepare_csv")
        # la tua funzione accetta un path string — passiamo la stringa
        blasto_labels_df = merge_sheets_and_prepare_csv(str(path_input_excel))
        logging.info("Dataframe creato. Righe: %d, Colonne: %d", len(blasto_labels_df), len(blasto_labels_df.columns))

        # 2) match double dishes e salvataggio CSV con patient_id
        logging.info("Step 2/3 — match_double_dishes_and_create_csv")
        if SAVE_OUTPUT:
            # signature: match_double_dishes_and_create_csv(path_double_dish_excel, blasto_labels_df, output_path_added_ID)
            match_double_dishes_and_create_csv(str(path_double_dish), blasto_labels_df, str(path_added_id))
            logging.info("CSV con patient_id salvato in: %s", path_added_id)
        else:
            logging.info("SAVE_OUTPUT=False -> salto salvataggio CSV con patient_id")

        # 3) filtro valid wells (opzionale)
        if APPLY_FILTER:
            if not path_valid_wells.exists():
                logging.warning("APPLY_FILTER=True ma file valid_wells non trovato: %s. Skip filtro.", path_valid_wells)
            else:
                logging.info("Step 3/3 — filter_by_valid_wells")
                # filter_by_valid_wells(final_table_path: str, valid_csv_path: str, output_path: str)
                # final_table_path useremo quello salvato sopra (path_added_id)
                if not SAVE_OUTPUT:
                    # se non abbiamo salvato, passiamo comunque il dataframe in memoria: salviamolo temporaneamente per il filtro
                    tmp_added = str(path_added_id)
                    blasto_labels_df.to_csv(tmp_added, index=False)
                    logging.info("Salvato temporaneamente %s per poter applicare il filtro.", tmp_added)

                filter_by_valid_wells(final_table_path=str(path_added_id),
                                      valid_csv_path=str(path_valid_wells),
                                      output_path=str(filtered_blasto_dataset))
                logging.info("Tabella filtrata salvata in: %s", filtered_blasto_dataset)
        else:
            logging.info("APPLY_FILTER=False -> skip filtro valid_wells")

        elapsed = time.time() - start_time
        logging.info("Preprocessing completato con successo in %.2f s", elapsed)

    except Exception as e:
        logging.exception("Errore nell'esecuzione del main: %s", e)
        raise
