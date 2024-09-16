import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sktime.classification.kernel_based import RocketClassifier
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

from config import Config_03_train_rocket as conf


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
    accuracy_dict = {}
    best_accuracy = 0
    best_kernel = None
    best_model_path = None

    for kernel in conf.kernels:
        conf.kernels = kernel

        # Carica i dati normalizzati dal file CSV
        df = load_normalized_data(conf.data_path)

        # Prepara i data loader
        X_train, X_test, y_train, y_test = prepare_data_loaders(df, conf.test_size)

        # Definisce il modello RocketClassifier
        model = RocketClassifier(num_kernels = kernel, random_state = conf.seed_everything(conf.seed), n_jobs=-1)

        # Inizia una nuova run su W&B
        wandb.init(
            project=conf.project_name,
            config={
                "exp_name": conf.exp_name + "," + str(kernel),
                "dataset": conf.dataset,
                "model": "RocketClassifier",
                "img_size": conf.img_size,
                "num_classes": conf.num_classes
            }
        )
        wandb.run.name = f"{conf.exp_name}_kernels_{kernel}"

        # Addestramento del modello
        model = train_model(model, X_train, y_train)

        # Valutazione del modello su train e test set
        train_accuracy, train_report = evaluate_model(model, X_train, y_train)
        test_accuracy, test_report = evaluate_model(model, X_test, y_test)

        # Log dei risultati su W&B
        log_results(wandb, train_accuracy, test_accuracy)

        # Salva l'accuratezza del test nel dizionario
        accuracy_dict[kernel] = test_accuracy

        # Stampa dei risultati
        print(f'Train Accuracy: {train_accuracy}')
        print(f'Test Accuracy: {test_accuracy}')
        print(f"Train Classification Report with {kernel} kernels:")
        print(train_report)
        print(f"Test Classification Report with {kernel} kernels:")
        print(test_report)

        # Aggiorna il modello migliore se l'accuratezza sul test è la migliore trovata finora
        if accuracy_dict[kernel] > best_accuracy:
            best_accuracy = accuracy_dict[kernel]
            best_kernel = kernel
            best_model_path = os.path.join(parent_dir, conf.test_dir, f"rocket_classifier_model_{kernel}_kernels.pkl")
            joblib.dump(model, best_model_path)
            print(f'Modello salvato in: {best_model_path}')

        # Fine della run W&B
        wandb.finish()

    # Stampa il dizionario delle accuratezze
    print("Accuratezza su test set per ogni kernel:", accuracy_dict)

    # Stampa il modello con la migliore accuratezza
    print(f"Il modello migliore è con {best_kernel} kernel, con un'accuratezza del {best_accuracy:.4f}")
    print(f"Modello salvato in: {best_model_path}")

    # Carica e stampa le metriche per il modello migliore
    best_model = joblib.load(best_model_path)
    X_train, X_test, y_train, y_test = prepare_data_loaders(load_normalized_data(conf.data_path), conf.test_size)
    _, best_test_report = evaluate_model(best_model, X_test, y_test)
    print('Best Model Test Classification Report:')
    print(best_test_report)



if __name__ == "__main__":
    # Misura il tempo di esecuzione della funzione main()
    execution_time = timeit.timeit(lambda: main(), number=1)
    print(f"Tempo impiegato per l'esecuzione di Rocket con vari kernel:", execution_time, "secondi")