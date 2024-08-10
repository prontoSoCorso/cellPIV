import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sktime.classification.deep_learning import LSTMFCNClassifier
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score, cohen_kappa_score, brier_score_loss
import joblib  # Per il salvataggio del modello
import timeit
import optuna

# Configurazione dei percorsi e dei parametri
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_03_train_lstmfcn_with_optuna as conf

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


def objective(trial):
    # Definisce lo spazio di ricerca degli iperparametri
    num_epochs = conf.num_epochs
    #num_epochs = trial.suggest_int('num_epochs', conf.min_num_epochs, conf.max_num_epochs)
    batch_size = trial.suggest_categorical('batch_size', conf.batch_size)
    dropout = trial.suggest_float('dropout', conf.min_dropout, conf.max_dropout, step=0.05)
    kernel_sizes = trial.suggest_categorical('kernel_sizes', conf.kernel_sizes)
    filter_sizes = trial.suggest_categorical('filter_sizes', conf.filter_sizes)
    lstm_size = trial.suggest_int('lstm_size', conf.min_lstm_size, conf.max_lstm_size)
    attention = conf.attention
    random_state = conf.seed_everything(conf.seed)
    verbose = conf.verbose

    # Carica i dati normalizzati dal file CSV
    df = load_normalized_data(conf.data_path)

    # Prepara i data loader
    X_train, X_val, y_train, y_val = prepare_data_loaders(df, conf.val_size)

    # Definisce il modello LSTMFCNClassifier con attenzione
    model = LSTMFCNClassifier(n_epochs=num_epochs, 
                              batch_size=batch_size,
                              dropout=dropout,
                              kernel_sizes=kernel_sizes,
                              filter_sizes=filter_sizes,
                              lstm_size=lstm_size,
                              attention=attention,
                              random_state=random_state,
                              verbose=verbose
                              )

    
    # Addestramento del modello
    model = train_model(model, X_train, y_train)

    # Valutazione del modello su train e val set
    val_metrics = evaluate_model(model, X_val, y_val)

    # Stampa dei risultati
    print(f'val Accuracy: {val_metrics[0]}')
    print('val Classification Report:')
    print(val_metrics[4])

    return val_metrics[0]


def main():
    study = optuna.create_study(direction='maximize', sampler=conf.sampler, pruner=conf.pruner)
    study.optimize(objective, n_trials=100)

    print(f'Numero di trial: {len(study.trials)}')
    print(f'Miglior trial: {study.best_trial.params}')

    # Salvataggio del modello
    model_save_path = os.path.join(parent_dir, "_04_test", "lstmfcnWithOptuna_classifier_model.pkl")
    joblib.dump(study, model_save_path)
    print(f'Modello salvato in: {model_save_path}')



if __name__ == "__main__":
    # Misura il tempo di esecuzione della funzione main()
    execution_time = timeit.timeit(main, number=1)
    print("Tempo impiegato per l'esecuzione di LSTMFCN:", execution_time, "secondi")
