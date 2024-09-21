import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sktime.classification.hybrid import HIVECOTEV2
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, brier_score_loss, confusion_matrix
import wandb
import joblib  # Per il salvataggio del modello
import timeit
import seaborn as sns
import matplotlib.pyplot as plt

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

# Funzione per valutare il modello con metriche estese
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]  # Probabilità della classe positiva
    accuracy = accuracy_score(y, y_pred)
    balanced_accuracy = balanced_accuracy_score(y, y_pred)
    kappa = cohen_kappa_score(y, y_pred)
    brier = brier_score_loss(y, y_prob, pos_label=1)
    cm = confusion_matrix(y, y_pred)
    return accuracy, balanced_accuracy, kappa, brier, cm

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
    csv_file_path = conf.data_path

    # Carica i dati normalizzati dal file CSV
    df = load_normalized_data(csv_file_path)

    # Prepara i data loader
    X_train, X_test, y_train, y_test = prepare_data_loaders(df, conf.test_size)

    # Definisce il modello HIVECOTEV2Classifier
    model = HIVECOTEV2(random_state=conf.seed_everything(conf.seed), n_jobs=-1)

    # Addestramento del modello
    model = train_model(model, X_train, y_train)

    # Valutazione del modello su train e test set
    train_metrics = evaluate_model(model, X_train, y_train)
    test_metrics = evaluate_model(model, X_test, y_test)

    # Stampa dei risultati
    print(f'Train Accuracy: {train_metrics[0]}')
    print(f'Train Balanced Accuracy: {train_metrics[1]}')
    print(f"Train Cohen's Kappa: {train_metrics[2]}")
    print(f'Train Brier Score Loss: {train_metrics[3]}')
    print(f'Test Accuracy: {test_metrics[0]}')
    print(f'Test Balanced Accuracy: {test_metrics[1]}')
    print(f"Test Cohen's Kappa: {test_metrics[2]}")
    print(f'Test Brier Score Loss: {test_metrics[3]}')

    # Salva la matrice di confusione per il test set
    save_confusion_matrix(test_metrics[4], f"confusion_matrix_hivecotev2.png")

    # Salvataggio del modello
    model_save_path = os.path.join(parent_dir, conf.test_dir, "hivecotev2_classifier_model.pkl")
    joblib.dump(model, model_save_path)
    print(f'Modello salvato in: {model_save_path}')


if __name__ == "__main__":
    # Misura il tempo di esecuzione della funzione main()
    execution_time = timeit.timeit(main, number=1)
    print("Tempo impiegato per l'esecuzione di HiveCoteV2:", execution_time, "secondi")
