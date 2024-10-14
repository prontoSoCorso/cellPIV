import sys
import os
import pandas as pd
from sktime.classification.kernel_based import RocketClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, brier_score_loss, confusion_matrix, f1_score
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

from config import Config_03_train_rocket as conf

# Funzione per caricare i dati normalizzati da CSV
def load_normalized_data(csv_file_path):
    return pd.read_csv(csv_file_path)

# Funzione per addestrare il modello
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

'''
# Funzione per preparare i data loader (in questo caso utilizziamo sklearn)
def prepare_data_loaders(df, test_size=0.3):
    X = df.iloc[:, 3:].values
    y = df['BLASTO NY'].values

    # Splitting the data into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=conf.seed_everything(conf.seed))

    return X_train, X_test, y_train, y_test
'''
    
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

    for kernel in conf.kernels:
        conf.kernels = kernel

        # Carica i dati normalizzati dal file CSV per il training
        df_train = load_normalized_data(conf.data_path)
        X_train = df_train.iloc[:, 3:].values
        y_train = df_train['BLASTO NY'].values

        # Carica i dati normalizzati dal file CSV per il test
        df_test = load_normalized_data(conf.test_path)
        X_test = df_test.iloc[:, 3:].values
        y_test = df_test['BLASTO NY'].values

        # Definisce il modello RocketClassifier
        model = RocketClassifier(num_kernels=kernel, random_state=conf.seed_everything(conf.seed), n_jobs=-1)

        # Addestramento del modello
        model = train_model(model, X_train, y_train)

        # Valutazione del modello su train e test set
        train_metrics = evaluate_model(model, X_train, y_train)
        test_metrics = evaluate_model(model, X_test, y_test)

        # Salva l'accuratezza del test nel dizionario
        accuracy_dict[kernel] = test_metrics[0]

        # Stampa dei risultati per il train set
        print(f'=====RESULTS WITH {kernel} KERNELS=====')
        print(f'Train Accuracy with {kernel} kernels: {train_metrics[0]}')
        print(f'Train Balanced Accuracy with {kernel} kernels: {train_metrics[1]}')
        print(f"Train Cohen's Kappa with {kernel} kernels: {train_metrics[2]}")
        print(f'Train Brier Score Loss with {kernel} kernels: {train_metrics[3]}')
        print(f'Train F1 Score with {kernel} kernels: {train_metrics[4]}')

        # Stampa dei risultati per il test set
        print(f'Test Accuracy with {kernel} kernels: {test_metrics[0]}')
        print(f'Test Balanced Accuracy with {kernel} kernels: {test_metrics[1]}')
        print(f"Test Cohen's Kappa with {kernel} kernels: {test_metrics[2]}")
        print(f'Test Brier Score Loss with {kernel} kernels: {test_metrics[3]}')
        print(f'Test F1 Score with {kernel} kernels: {test_metrics[4]}')

        '''
        # Salva la matrice di confusione per il test set
        save_confusion_matrix(test_metrics[5], f'confusion_matrix_{kernel}_kernels.png')  # Salva come immagine
        '''

        # Aggiorna il modello migliore se l'accuratezza sul test è la migliore trovata finora
        if accuracy_dict[kernel] > best_accuracy:
            best_accuracy = accuracy_dict[kernel]
            best_kernel = kernel
            best_model_path = os.path.join(parent_dir, conf.test_dir, f"rocket_classifier_model_{kernel}_kernels.pkl")
            joblib.dump(model, best_model_path)
            print(f'Modello salvato in: {best_model_path}')

    # Stampa il dizionario delle accuratezze
    print("Accuratezza su test set per ogni kernel:", accuracy_dict)

    # Stampa il modello con la migliore accuratezza
    print(f"Il modello migliore è con {best_kernel} kernel, con un'accuratezza del {best_accuracy:.4f}")
    print(f"Modello salvato in: {best_model_path}")

    # Carica e stampa le metriche per il modello migliore
    best_model = joblib.load(best_model_path)
    df_train = load_normalized_data(conf.data_path)
    df_test = load_normalized_data(conf.test_path)
    X_train = df_train.iloc[:, 3:].values
    y_train = df_train['BLASTO NY'].values
    X_test = df_test.iloc[:, 3:].values
    y_test = df_test['BLASTO NY'].values
    
    # Valutazione del modello migliore
    _, _, _, _, _, best_test_cm = evaluate_model(best_model, X_test, y_test)  # Estrarre solo la matrice di confusione
    
    # Salva la matrice di confusione per il miglior modello
    save_confusion_matrix(best_test_cm, f"confusion_matrix_rocket_{best_kernel}_kernels.png")


if __name__ == "__main__":
    # Misura il tempo di esecuzione della funzione main()
    execution_time = timeit.timeit(lambda: main(), number=1)
    print(f"Tempo impiegato per l'esecuzione di Rocket con vari kernel:", execution_time, "secondi")
