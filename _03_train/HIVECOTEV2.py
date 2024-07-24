import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sktime.classification.hybrid import HIVECOTEV2
from sklearn.metrics import accuracy_score, classification_report
import wandb
import joblib  # Per il salvataggio del modello
import timeit

# Configurazione dei percorsi e dei parametri
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_03_train_hivecote as conf  # Aggiorna questo import se necessario


# Funzione per caricare i dati normalizzati da CSV
def load_normalized_data(csv_file_path):
    return pd.read_csv(csv_file_path)

# Funzione per preparare i data loader (in questo caso utilizziamo sklearn)
def prepare_data_loaders(df, test_size=0.3):
    X = df.iloc[:, 3:].values
    y = df['BLASTO NY'].values

    # Splitting the data into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=conf.seed_everything(conf.seed))

    return X_train, X_test, y_train, y_test

# Funzione per addestrare il modello
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

# Funzione per valutare il modello
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, target_names=["Class 0", "Class 1"])
    return accuracy, report

# Funzione per loggare i risultati su W&B
def log_results(wandb, train_accuracy, test_accuracy):
    wandb.log({
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy
    })



def main():
    csv_file_path = conf.data_path

    # Carica i dati normalizzati dal file CSV
    df = load_normalized_data(csv_file_path)

    # Prepara i data loader
    X_train, X_test, y_train, y_test = prepare_data_loaders(df, conf.test_size)

    # Definisce il modello HIVECOTEV2Classifier
    model = HIVECOTEV2(random_state=conf.seed_everything(conf.seed), n_jobs=-1)

    # Inizia una nuova run su W&B
    wandb.init(
        project=conf.project_name,
        config={
            "exp_name": conf.exp_name,
            "dataset": conf.dataset,
            "model": "HIVECOTEV2Classifier",
            "img_size": conf.img_size,
            "num_classes": conf.num_classes
        }
    )
    wandb.run.name = conf.exp_name

    # Addestramento del modello
    model = train_model(model, X_train, y_train)

    # Valutazione del modello su train e test set
    train_accuracy, train_report = evaluate_model(model, X_train, y_train)
    test_accuracy, test_report = evaluate_model(model, X_test, y_test)

    # Log dei risultati su W&B
    log_results(wandb, train_accuracy, test_accuracy)

    # Stampa dei risultati
    print(f'Train Accuracy: {train_accuracy}')
    print(f'Test Accuracy: {test_accuracy}')
    print('Train Classification Report:')
    print(train_report)
    print('Test Classification Report:')
    print(test_report)

    # Salvataggio del modello
    model_save_path = os.path.join(parent_dir, "_04_test", "hivecote_classifier_model.pkl")
    joblib.dump(model, model_save_path)
    print(f'Modello salvato in: {model_save_path}')

    # Fine della run W&B
    wandb.finish()


if __name__ == "__main__":
    # Misura il tempo di esecuzione della funzione main()
    execution_time = timeit.timeit(main, number=1)
    print("Tempo impiegato per l'esecuzione di HiveCoteV2:", execution_time, "secondi")
