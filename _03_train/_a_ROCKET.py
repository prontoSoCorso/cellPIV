import sys
import os
import pandas as pd
from sktime.classification.kernel_based import RocketClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, brier_score_loss, confusion_matrix, f1_score
import timeit
import seaborn as sns
import matplotlib.pyplot as plt
import torch

# Configurazione dei percorsi e dei parametri
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_03_train as conf

# Funzione per caricare i dati normalizzati da CSV
def load_data(csv_file_path):
    return pd.read_csv(csv_file_path)

# Funzione per addestrare il modello
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

# Funzione per valutare il modello con metriche estese
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]  # Probabilità della classe positiva
    accuracy = accuracy_score(y, y_pred)
    balanced_accuracy = balanced_accuracy_score(y, y_pred)
    kappa = cohen_kappa_score(y, y_pred)
    brier = brier_score_loss(y, y_prob, pos_label=1)
    f1 = f1_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    return accuracy, balanced_accuracy, kappa, brier, f1, cm

# Funzione per salvare la matrice di confusione come immagine
def save_confusion_matrix(cm, filename):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False, xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(filename)
    plt.close()

def main():
    accuracy_dict = {}
    best_accuracy = 0
    best_kernel = None
    best_model_path = None

    # Specifica il numero di giorni desiderati
    selected_days = "3Days"

    # Ottieni i percorsi dal config
    train_path, val_path, test_path = conf.get_paths(selected_days)

    # Carico i dati normalizzati
    df_train = load_data(train_path)
    df_val = load_data(val_path)
    df_test = load_data(test_path)

    X_train = df_train.iloc[:, 3:].values  # Le colonne da 3 in poi contengono la serie temporale
    y_train = df_train['BLASTO NY'].values  # Colonna target

    X_val = df_val.iloc[:, 3:].values  # Le colonne da 3 in poi contengono la serie temporale
    y_val = df_val['BLASTO NY'].values  # Colonna target

    X_test = df_test.iloc[:, 3:].values  # Le colonne da 3 in poi contengono la serie temporale
    y_test = df_test['BLASTO NY'].values  # Colonna target

    for kernel in conf.kernels:
        conf.kernels = kernel

        # Definisce il modello RocketClassifier
        model = RocketClassifier(num_kernels=kernel, random_state=conf.seed_everything(conf.seed), n_jobs=-1)

        # Addestramento del modello
        model = train_model(model, X_train, y_train)

        # Valutazione del modello su train e test set
        train_metrics = evaluate_model(model, X_train, y_train)
        val_metrics = evaluate_model(model, X_val, y_val)
        
        # Salva l'accuratezza del test nel dizionario
        accuracy_dict[kernel] = val_metrics[0]

        # Stampa dei risultati per il train set
        print(f'=====RESULTS WITH {kernel} KERNELS=====')
        print(f'Train Accuracy with {kernel} kernels: {train_metrics[0]}')
        print(f'Train Balanced Accuracy with {kernel} kernels: {train_metrics[1]}')
        print(f"Train Cohen's Kappa with {kernel} kernels: {train_metrics[2]}")
        print(f'Train Brier Score Loss with {kernel} kernels: {train_metrics[3]}')
        print(f'Train F1 Score with {kernel} kernels: {train_metrics[4]}')

        # Stampa dei risultati per il test set
        print(f'Val Accuracy with {kernel} kernels: {val_metrics[0]}')
        print(f'Val Balanced Accuracy with {kernel} kernels: {val_metrics[1]}')
        print(f"Val Cohen's Kappa with {kernel} kernels: {val_metrics[2]}")
        print(f'Val Brier Score Loss with {kernel} kernels: {val_metrics[3]}')
        print(f'Val F1 Score with {kernel} kernels: {val_metrics[4]}')

        # Aggiorna il modello migliore se l'accuratezza sul test è la migliore trovata finora
        if accuracy_dict[kernel] > best_accuracy:
            best_accuracy = accuracy_dict[kernel]
            best_kernel = kernel

    # Stampa il dizionario delle accuratezze
    print("Accuratezza su test set per ogni kernel:", accuracy_dict)

    # Stampa il modello con la migliore accuratezza
    print(f"Il modello migliore è con {best_kernel} kernel, con un'accuratezza del {best_accuracy:.4f}")

    # Rialleno il modello su tutti i dati con il numero di kernel migliore che ho trovato e lo salvo
    # Combina i due DataFrame in un unico DataFrame
    df = pd.concat([df_train, df_val], ignore_index=True)
    
    X = df.iloc[:, 3:].values  # Le colonne da 3 in poi contengono la serie temporale
    y = df['BLASTO NY'].values  # Colonna target

    model = RocketClassifier(num_kernels=best_kernel, random_state=conf.seed_everything(conf.seed), n_jobs=-1)
    model = train_model(model, X, y)

    # Valutazione finale sul test set
    accuracy, balanced_accuracy, kappa, brier, f1, cm = evaluate_model(model, X_test, y_test)
    print("\n=====FINAL TEST RESULTS=====")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"Test Cohen's Kappa: {kappa:.4f}")
    print(f"Test Brier Score Loss: {brier:.4f}")
    print(f"Test F1 Score: {f1:.4f}")

    cm_path = os.path.join(parent_dir, "confusion_matrix_ROCKET_" + selected_days + ".png")
    save_confusion_matrix(cm, cm_path)

    best_model_path = os.path.join(parent_dir, conf.test_dir, f"best_rocket_model_{selected_days}.pth")
    torch.save(model, best_model_path)
    print(f'Modello salvato in: {best_model_path}')
    

if __name__ == "__main__":
    # Misura il tempo di esecuzione della funzione main()
    execution_time = timeit.timeit(lambda: main(), number=1)
    print(f"Tempo impiegato per l'esecuzione di Rocket con vari kernel:", execution_time, "secondi")