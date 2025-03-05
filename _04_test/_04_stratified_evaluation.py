import warnings
warnings.filterwarnings("ignore", message="CUDA initialization: CUDA unknown error")
warnings.filterwarnings("ignore", message="torch.meshgrid:")
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

import sys
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch
from tabulate import tabulate
import torch.multiprocessing as mp
import time
import joblib
import matplotlib as plt
import seaborn as sns

# Aggiungo il percorso del progetto al sys.path
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = current_dir
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_03_train as conf
from _03_train._c_ConvTranUtils import CustomDataset
import _04_test._myFunctions as _myFunctions


def main(merge_type, days_to_consider = 1):
    device = conf.device
    mp.set_start_method("spawn", force=True)  # Imposta multiprocessing con spawn

    # ---------------------------
    # 1. Caricamento e Preprocessing del DB
    # ---------------------------
    db_file = os.path.join(parent_dir, "DB morpheus UniPV.xlsx")
    df_db = pd.read_excel(db_file)

    # Verifica che le colonne di interesse siano presenti (es. "slide_well" e "PN")
    if "slide_well" not in df_db.columns or "PN" not in df_db.columns:
        raise ValueError("Le colonne 'slide_well' e 'PN' devono essere presenti nel file DB morpheus UniPV.xlsx.")

    table1 = df_db.groupby("PN")["blasto ny"].value_counts().unstack()
    print(tabulate(table1, headers="keys", tablefmt="pretty"))

    '''
    Le diverse tipologie di pronuclei sono:
    - PN = 0 (Apronucleato, fecondazione fallita)
    - PN = 1 (Monopronucleato, L'embrione ha un solo pronucleo invece dei due attesi--> anomalo, mancato sviluppo del pronucleo maschile o femminile)
    - PN = 1.1 (Probabilmente Monopronucleato con caratteristiche particolari, Simile al caso 1PN, ma potrebbe avere una morfologia particolare del pronucleo)
    - PN = 2 (Dipropronucleato, normale)
    - PN = 2.1 (Probabilmente indica una variante di embrioni 2PN con caratteristiche morfologiche particolari, che potrebbero influenzarne lo sviluppo)
    - PN = 3 (Tripronucleato, probabile polispermia o problemi nella estrusione del secondo globulo polare. Generalmente non trasferiti perché scarsissima possibilità sviluppo sano)
    - PN = >3 (Polipronucleato, anomalo, disfunzionale)
    - PN = deg (embrioni non vitali, con severi danni cellulari, no divisione cellulare o eccessiva frammentazione)
    '''
    
    if merge_type == "anomalous":
        # Fondo "2.1PN", "1.1PN", "1PN", "3PN", ">3PN" in una nuova categoria (Sono tutti i fecondati anomali)
        df_db["merged_PN"] = df_db["PN"].replace({"1PN":"anomalous", "1.1PN":"anomalous", "2.1PN":"anomalous", "3PN":"anomalous", ">3PN":"anomalous"})
        table2 = df_db.groupby("merged_PN")["blasto ny"].value_counts().unstack()
        print(tabulate(table2, headers="keys", tablefmt="pretty"))
    elif merge_type == "not_vital":
        # Fondo "0PN" e "deg" in una nuova categoria (non vitali)
        df_db["merged_PN"] = df_db["PN"].replace({"0PN":"not_vital", "deg":"not_vital"})
        table2 = df_db.groupby("merged_PN")["blasto ny"].value_counts().unstack()
        print(tabulate(table2, headers="keys", tablefmt="pretty"))
    elif merge_type == "no_merging":
        # No merging
        print("=========== No selection of merge type ===========")
    else:
        print(f"No valid merge type selected")
        exit()

    output_csv_first_part = f"stratified_model_performance_{merge_type}"


    # ---------------------------
    # 2. Caricamento del Test CSV
    # ---------------------------
    # Scelgo numero di giorni per il test
    test_csv_file = os.path.join(parent_dir, f"Normalized_sum_mean_mag_{days_to_consider}Days_test.csv")
    if not os.path.exists(test_csv_file):
        raise FileNotFoundError(f"Modello ROCKET non trovato in {test_csv_file}")
    df_test = pd.read_csv(test_csv_file)

    # Assumiamo che df_test contenga la colonna "dish_well" per effettuare il merge
    if "dish_well" not in df_test.columns:
        raise ValueError("La colonna 'dish_well' deve essere presente nel file di test.")

    # ---------------------------
    # 3. Merge dei Dati
    # ---------------------------
    # Uniamo il DB e il test usando:
    #   DB: "slide_well" (che contiene il riferimento al well)
    #   Test: "dish_well"
    df_merged = pd.merge(df_test, df_db[["slide_well", "merged_PN"]], left_on="dish_well", right_on="slide_well", how="left")
    # Se alcune righe non trovano match, possiamo eliminarle (o gestirle in altro modo)
    df_merged = df_merged.dropna(subset=["merged_PN"])
    print(f"Numero di istanze dopo merge: {df_merged.shape[0]}")

    # ---------------------------
    # 4. Stratificazione in base a merged_PN
    # ---------------------------
    # La stratificazione servirà per valutare le prestazioni per ciascun gruppo.
    strata = df_merged["merged_PN"]

    # Per comodità, salviamo l'elenco dei livelli
    pn_categories = df_merged["merged_PN"].unique()
    print("Categorie PN dopo merge:", pn_categories)

    # ---------------------------
    # 5. Estrazione delle Features e Target
    # ---------------------------
    # Suppongo che la colonna "BLASTO NY" sia il target e che i temporal values abbiano nome colonna del tipo "value_" 
    temporal_columns = [col for col in df_merged.columns if col.startswith("value_")]
    X = df_merged[temporal_columns].values  # adatta l'indice se necessario
    y = df_merged["BLASTO NY"].values






    # ---------------------------
    # 6. Caricamento dei Modelli Pre-addestrati e creazione funzione per valutazione su test
    # ---------------------------
    # Importa i modelli


    # --- ROCKET ---
    rocket_model_path = os.path.join(current_dir, f"best_rocket_model_{days_to_consider}Days.joblib")
    if not os.path.exists(rocket_model_path):
        raise FileNotFoundError(f"Modello ROCKET non trovato in {rocket_model_path}")
    #rocket_model = torch.load(rocket_model_path, weights_only=False)
    # Caricamento con soglia ottimale
    rocket_artifact = joblib.load(rocket_model_path)
    rocket_model = rocket_artifact["classifier"]
    rocket_transformer = rocket_artifact["rocket"] 
    final_threshold = rocket_artifact["final_threshold"]
    num_kernels = rocket_artifact["num_kernels"]



    # --- LSTMFCN ---
    from _03_train._b_LSTMFCN import LSTMFCN
    lstm_model_path = os.path.join(current_dir, f"best_lstmfcn_model_{days_to_consider}Days.pth")
    if not os.path.exists(lstm_model_path):
        raise FileNotFoundError(f"Modello LSTMFCN non trovato in {lstm_model_path}")
    
    # Caricamento con soglia ottimale
    lstm_checkpoint = torch.load(lstm_model_path, map_location=device, weights_only=False)
    lstm_model = LSTMFCN(
        lstm_size=conf.lstm_size_FCN,
        filter_sizes=conf.filter_sizes_FCN,
        kernel_sizes=conf.kernel_sizes_FCN,
        dropout=conf.dropout_FCN,
        num_layers=conf.num_layers_FCN
    ).to(device)
        
    lstm_model.load_state_dict(lstm_checkpoint['model_state_dict'])
    lstm_threshold = lstm_checkpoint.get('best_threshold', 0.5)

    # Per LSTMFCN, funzione di test che lavora con DataLoader:
    def test_model_LSTMFCN(model, X, y, threshold, batch_size=conf.batch_size_FCN):
        from torch.utils.data import TensorDataset, DataLoader
        # Prepara i dati: aggiunge dimensione canale se necessario
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        y_tensor = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=conf.batch_size_FCN, shuffle=False)
        
        model.eval()
        all_pred, all_prob, all_true = [], [], []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)

                prob = torch.softmax(output, dim=1)[:, 1]
                pred = (prob >= threshold).int()

                all_true.extend(y_batch.cpu().numpy().flatten())
                all_pred.extend(pred.cpu().numpy().flatten())
                all_prob.extend(prob.cpu().numpy().flatten())

        return np.array(all_true), np.array(all_pred), np.array(all_prob)




    # --- ConvTran ---
    from _99_ConvTranModel.model import model_factory
    from _99_ConvTranModel.utils import load_model
    conv_model_path = os.path.join(current_dir, f"best_convTran_model_{days_to_consider}Days.pkl")
    if not os.path.exists(conv_model_path):
        raise FileNotFoundError(f"Modello ConvTran non trovato in {conv_model_path}")
    
    
    # Funzione per il caricamento del modello
    def load_model_with_threshold(model, path, device):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint.get('best_threshold', 0.5)
    
    conf.num_labels = 2     # len(set(loader.dataset.labels)) --> lo forzo a 2 perché le classi sono due, anceh se alcune volte il test non ha esempi positivi (e.g per i deg o quelli con 0PN non ci sono mai blasto)
    if X.ndim == 2:
        conf.Data_shape = (1, X.shape[1])
    elif X.ndim == 3:
        conf.Data_shape = (X.shape[0], X.shape[2])
    else:
        raise ValueError("x deve avere 2 o 3 dimensioni.")
    # è la shape dei dati, del tipo: (num_channels, num_temporal_steps)

    # Caricamento modello e soglia
    conv_model = model_factory(conf).to(device)
    conv_model, conv_threshold = load_model_with_threshold(conv_model, conv_model_path, device)
    
    
    # Funzione di test per ConvTran (simile a quella usata precedentemente):
    def test_model_ConvTran(X, y, threshold, batch_size=conf.batch_size):
        dataset = CustomDataset(X.reshape(X.shape[0], 1, -1), y)
        num_workers = max(1, os.cpu_count() - 6)  # Lascio almeno 6 core per il sistema
        loader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        conv_model.eval()
        all_pred, all_prob, all_true = [], [], []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = conv_model(X_batch)

                prob = torch.softmax(output, dim=1)[:, 1]
                pred = (prob >= threshold).int()

                all_true.extend(y_batch.cpu().numpy().flatten())
                all_pred.extend(pred.cpu().numpy().flatten())
                all_prob.extend(prob.cpu().numpy().flatten())

        return np.array(all_true), np.array(all_pred), np.array(all_prob)








    # ---------------------------
    # 7. Valutazione dei Modelli sul Test Stratificato
    # ---------------------------

    # Creiamo un dizionario per salvare i risultati
    performance_results = []

    def appendResults(model_name, group, metrics):
        performance_results.append({
            "Model": model_name,
            "Stratum": group,
            "Balanced Accuracy": metrics["balanced_accuracy"],
            "F1 Score": metrics["f1"]
        })



    # --- Valutazione ROCKET ---
    print("\nValutazione modello ROCKET...")
    y_pred_grp, y_prob_grp = _myFunctions.test_model_ROCKET(transformer=rocket_transformer, 
                                                            model=rocket_model, 
                                                            X=X, 
                                                            threshold=final_threshold)
    metrics_rocket = _myFunctions.calculate_metrics(y, y_pred_grp, y_prob_grp)
    appendResults(model_name="ROCKET", group="Overall", metrics=metrics_rocket)
    
    # Valutazione per ciascun gruppo PN
    for group, group_df in df_merged.groupby("merged_PN"):
        X_group = group_df[temporal_columns].values
        y_group = group_df["BLASTO NY"].values
        y_pred_grp, y_prob_grp = _myFunctions.test_model_ROCKET(transformer=rocket_transformer, 
                                                                model=rocket_model, 
                                                                X=X_group,
                                                                threshold=final_threshold)
        metrics_rocket_grp = _myFunctions.calculate_metrics(y_group, y_pred_grp, y_prob_grp)
        appendResults(model_name="ROCKET", group=group, metrics=metrics_rocket_grp)
        


    # --- Valutazione LSTMFCN ---
    print("\nValutazione modello LSTMFCN...")
    _, y_pred_l, y_prob_l = test_model_LSTMFCN(model=lstm_model, X=X, y=y, threshold=lstm_threshold)
    metrics_lstmfcn = _myFunctions.calculate_metrics(y, y_pred_l, y_prob_l)
    appendResults(model_name="LSTMFCN", group="Overall", metrics=metrics_lstmfcn)
    
    for group, group_df in df_merged.groupby("merged_PN"):
        X_group = group_df[temporal_columns].values
        y_group = group_df["BLASTO NY"].values
        _, y_pred_grp, y_prob_grp = test_model_LSTMFCN(model=lstm_model, X=X_group, y=y_group, threshold=lstm_threshold)
        metrics_lstmfcn_grp = _myFunctions.calculate_metrics(y_group, y_pred_grp, y_prob_grp)
        appendResults(model_name="LSTMFCN", group=group, metrics=metrics_lstmfcn_grp)

    
    # --- Valutazione ConvTran ---
    print("\nValutazione modello ConvTran...")
    _, y_pred_c, y_prob_c = test_model_ConvTran(X=X, y=y, threshold=0.5)
    metrics_convTran = _myFunctions.calculate_metrics(y, y_pred_c, y_prob_c)
    appendResults(model_name="ConvTran", group="Overall", metrics=metrics_convTran)

    for group, group_df in df_merged.groupby("merged_PN"):
        X_group = group_df[temporal_columns].values
        y_group = group_df["BLASTO NY"].values
        _, y_pred_grp, y_prob_grp = test_model_ConvTran(X=X_group, y=y_group, threshold=0.5)
        metrics_convTran_grp = _myFunctions.calculate_metrics(y_group, y_pred_grp, y_prob_grp)
        appendResults(model_name="ConvTran", group=group, metrics=metrics_convTran_grp)
    




    # ---------------------------
    # 8. Salvataggio dei Risultati
    # ---------------------------
    df_performance = pd.DataFrame(performance_results)
    output_csv_complete = f"{output_csv_first_part}_{days_to_consider}Days.csv"
    output_file = os.path.join(current_dir, output_csv_complete)
    df_performance.to_csv(output_file, index=False)
    print(f"\n📁 Risultati della valutazione stratificata salvati in: {output_file}")

    print("\nEsecuzione completata.")


if __name__ == "__main__":
    merge_type = "not_vital"    # "anomalous" OR "not_vital" OR "no_merging"
    days_to_consider = 5        # 1,3,5,7
    start_time = time.time()
    main(merge_type, days_to_consider)
    print("Execution time:", time.time() - start_time, "seconds")
