import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sktime.classification.deep_learning import LSTMFCNClassifier
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score, cohen_kappa_score, brier_score_loss
import wandb
import joblib  # Per il salvataggio del modello
import timeit

# Configurazione dei percorsi e dei parametri
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_03_train_lstmfcn as conf

# Funzione per caricare i dati normalizzati da CSV
def load_normalized_data(csv_file_path):
    return pd.read_csv(csv_file_path)

# Funzione per preparare i data loader (in questo caso utilizziamo sklearn)
def prepare_data_loaders(df, val_size=0.3):
    X = df.iloc[:, 3:].values
    y = df['BLASTO NY'].values

    # Splitting the data into train, validation, and val sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=conf.seed_everything(conf.seed))

    return X_train, X_val, y_train, y_val

# Funzione per addestrare il modello
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

# Funzione per valutare il modello
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]  # Probabilità della classe positiva
    accuracy = accuracy_score(y, y_pred)
    balanced_accuracy = balanced_accuracy_score(y, y_pred)
    kappa = cohen_kappa_score(y, y_pred)
    brier = brier_score_loss(y, y_prob, pos_label=1)
    report = classification_report(y, y_pred, target_names=["Class 0", "Class 1"])
    return accuracy, balanced_accuracy, kappa, brier, report

# Funzione per loggare i risultati su W&B
def log_results(wandb, train_metrics, val_metrics):
    wandb.log({
        'train_accuracy': train_metrics[0],
        'train_balanced_accuracy': train_metrics[1],
        'train_kappa': train_metrics[2],
        'train_brier': train_metrics[3],
        'val_accuracy': val_metrics[0],
        'val_balanced_accuracy': val_metrics[1],
        'val_kappa': val_metrics[2],
        'val_brier': val_metrics[3]
    })


def main():
    csv_file_path = conf.data_path

    # Carica i dati normalizzati dal file CSV
    df = load_normalized_data(csv_file_path)

    # Prepara i data loader
    X_train, X_val, y_train, y_val = prepare_data_loaders(df, conf.val_size)

    # Definisce il modello LSTMFCNClassifier con attenzione
    model = LSTMFCNClassifier(n_epochs=conf.num_epochs, 
                              batch_size=conf.batch_size,
                              dropout=conf.dropout,
                              kernel_sizes=conf.kernel_sizes,
                              filter_sizes=conf.filter_sizes,
                              lstm_size=conf.lstm_size,
                              attention=conf.attention,
                              random_state=conf.seed,
                              verbose=conf.verbose
                              )

    # Inizia una nuova run su W&B
    wandb.init(
        project=conf.project_name,
        config={
            "exp_name": conf.exp_name,
            "dataset": conf.dataset,
            "model": "LSTMFCNClassifier",
            "img_size": conf.img_size,
            "num_classes": conf.num_classes,
            "num_epochs": conf.num_epochs,
            "kernel_sizes": conf.kernel_sizes,
            "filter_sizes": conf.filter_sizes,
            "lstm_size": conf.lstm_size,
            "attention": conf.attention,
            "random_state": conf.seed,
            "verbose": conf.verbose
        }
    )
    wandb.run.name = conf.exp_name

    # Addestramento del modello
    model = train_model(model, X_train, y_train)

    # Valutazione del modello su train e val set
    train_metrics = evaluate_model(model, X_train, y_train)
    val_metrics = evaluate_model(model, X_val, y_val)

    # Log dei risultati su W&B
    log_results(wandb, train_metrics, val_metrics)

    # Stampa dei risultati
    print(f'Train Accuracy: {train_metrics[0]}')
    print(f'val Accuracy: {val_metrics[0]}')
    print('Train Classification Report:')
    print(train_metrics[4])
    print('val Classification Report:')
    print(val_metrics[4])

    # Salvataggio del modello
    model_save_path = os.path.join(parent_dir, "_04_test", "lstmfcn_classifier_model.pkl")
    joblib.dump(model, model_save_path)
    print(f'Modello salvato in: {model_save_path}')

    # Fine della run W&B
    wandb.finish()


if __name__ == "__main__":
    # Misura il tempo di esecuzione della funzione main()
    execution_time = timeit.timeit(main, number=1)
    print("Tempo impiegato per l'esecuzione di LSTMFCN:", execution_time, "secondi")
