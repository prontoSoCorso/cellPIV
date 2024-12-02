import sys
import os
import pandas as pd
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, brier_score_loss, confusion_matrix, f1_score
import joblib  # Per il salvataggio del modello
import seaborn as sns
import matplotlib.pyplot as plt
import timeit

# Configurazione dei percorsi e dei parametri
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import paths_for_models as paths_for_models

# Funzione per caricare i dati normalizzati da CSV
def load_normalized_data(csv_file_path):
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
    # Carica i dati normalizzati dal file CSV per il training
    df_train = load_normalized_data(paths_for_models.data_path_train)
    df_val  = load_normalized_data(paths_for_models.data_path_val)
    
    df = pd.concat([df_train, df_val], ignore_index=True)
    
    X = df.iloc[:, 3:].values  # Le colonne da 3 in poi contengono la serie temporale
    y = df['BLASTO NY'].values  # Colonna target

    # Definisce il modello KNNClassifier
    model = KNeighborsTimeSeriesClassifier(n_neighbors=7, 
                                           weights='distance', #uniform di default, se no distance
                                           algorithm='brute',
                                           distance='dtw',
                                           distance_params=None,
                                           distance_mtype=None,
                                           pass_train_distances=False,
                                           leaf_size=30,
                                           n_jobs=-1)

    # Addestramento del modello
    model = train_model(model, X, y)
    print("=====Addestramento completato=====")

    # Carico dati normalizzati del test per il test e creo X e labels
    df_test = load_normalized_data(paths_for_models.test_path)
    X_test = df_test.iloc[:, 3:].values
    y_test = df_test['BLASTO NY'].values

    test_metrics = evaluate_model(model, X_test, y_test)

    # Risultati su test set
    print(f'Test Accuracy: {test_metrics[0]}')
    print(f'Test Balanced Accuracy: {test_metrics[1]}')
    print(f"Test Cohen's Kappa: {test_metrics[2]}")
    print(f'Test Brier Score Loss: {test_metrics[3]}')
    print(f'Test F1 Score: {test_metrics[4]}')
    
    save_confusion_matrix(test_metrics[5], "confusion_matrix_KNN.png")

    # Salvataggio del modello
    model_save_path = os.path.join(parent_dir, paths_for_models.test_dir, "KNN_classifier_model.pkl")
    joblib.dump(model, model_save_path)
    print(f'Modello salvato in: {model_save_path}')
    

if __name__ == "__main__":
    # Misura il tempo di esecuzione della funzione main()
    execution_time = timeit.timeit(lambda: main(), number=1)
    print(f"Tempo impiegato per l'esecuzione di KNN:", execution_time, "secondi")
