import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sktime.classification.deep_learning import LSTMFCNClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, balanced_accuracy_score, cohen_kappa_score, brier_score_loss
import joblib  # Per il salvataggio del modello
import timeit
import optuna
import seaborn as sns
import matplotlib.pyplot as plt

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

    # Stampa dei risultati con metriche estese
    print(f'Validation Accuracy: {val_metrics[0]}')
    print(f'Validation Balanced Accuracy: {val_metrics[1]}')
    print(f"Validation Cohen's Kappa: {val_metrics[2]}")
    print(f'Validation Brier Score Loss: {val_metrics[3]}')
    print(f'Validation F1 Score: {val_metrics[4]}')

    return val_metrics[0]


def main():
    study = optuna.create_study(direction='maximize', sampler=conf.sampler, pruner=conf.pruner)
    study.optimize(objective, n_trials=100)

    best_trial = study.best_trial
    print(f'Numero di trial: {len(study.trials)}')
    print(f'Migliore trial: {best_trial.number}, \nMigliore Accuracy: {best_trial.value}, \nBest trial params: {best_trial.params}')

    #Devo riaddestrare il modello migliore. Infatti Il best_trial che si ottiene da Optuna contiene solo i parametri ottimali che 
    # hanno prodotto la miglior performance, non il modello addestrato stesso.
    # Ricostruisco il modello usando i migliori parametri trovati
    df = load_normalized_data(conf.data_path)
    X_train, X_val, y_train, y_val = prepare_data_loaders(df, conf.val_size)
    best_model = LSTMFCNClassifier(n_epochs=best_trial.params['n_epochs'],
                                    batch_size=best_trial.params['batch_size'],
                                    dropout=best_trial.params['dropout'],
                                    kernel_sizes=best_trial.params['kernel_sizes'],
                                    filter_sizes=best_trial.params['filter_sizes'],
                                    lstm_size=best_trial.params['lstm_size'],
                                    attention=best_trial.params['attention'],
                                    random_state=conf.seed_everything(conf.seed),
                                    verbose=conf.verbose
                                    )
    
    best_model = train_model(best_model, X_train, y_train)

    # Valutazione del modello su train e val set
    val_metrics = evaluate_model(best_model, X_val, y_val)

    # Stampa dei risultati con metriche estese
    print(f'Validation Accuracy: {val_metrics[0]}')
    print(f'Validation Balanced Accuracy: {val_metrics[1]}')
    print(f"Validation Cohen's Kappa: {val_metrics[2]}")
    print(f'Validation Brier Score Loss: {val_metrics[3]}')
    print(f'Validation F1 Score: {val_metrics[4]}')

    # Salva la matrice di confusione per il validation set
    save_confusion_matrix(val_metrics[5], f'confusion_matrix_lstmfcn_.png')

    # Salvataggio del modello addestrato
    model_save_path = os.path.join(parent_dir, conf.test_dir, "lstmfcn_with_optuna_best_model.pkl")
    joblib.dump(best_model, model_save_path)
    print(f'Modello addestrato salvato in: {model_save_path}')

if __name__ == "__main__":
    # Misura il tempo di esecuzione della funzione main()
    execution_time = timeit.timeit(main, number=1)
    print("Tempo impiegato per l'esecuzione di LSTMFCN:", execution_time, "secondi")
