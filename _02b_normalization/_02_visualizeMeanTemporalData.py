import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Configurazione dei percorsi e dei parametri
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_02b_normalization as conf

# Funzione per creare il plot
def create_plot(blasto, no_blasto, title, filename):
    # Calcolo della media e della deviazione standard per ciascun gruppo
    blasto_mean = blasto.iloc[:, 3:].mean()
    blasto_std = blasto.iloc[:, 3:].std()

    no_blasto_mean = no_blasto.iloc[:, 3:].mean()
    no_blasto_std = no_blasto.iloc[:, 3:].std()

    # Metto sulle x i numerini e non i value_numero
    x = np.arange(1, len(blasto_mean) + 1)

    # Creazione del grafico
    plt.figure(figsize=(10, 6)) 
    if conf.temporalDataType == "vorticity_dict":
        plt.ylim(-0.1, 0.1)
    else:
        plt.ylim(-0.2,0.8)

    # blasto
    plt.plot(x, blasto_mean, label='Blasto', color='blue')
    plt.fill_between(x, blasto_mean - blasto_std, blasto_mean + blasto_std, color='blue', alpha=0.2)

    # no_blasto
    plt.plot(x, no_blasto_mean, label='No Blasto', color='red')
    plt.fill_between(x, no_blasto_mean - no_blasto_std, no_blasto_mean + no_blasto_std, color='red', alpha=0.2)

    # Plot e Save
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Optical Flow Metric')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    
    
    df_train = pd.read_csv(paths_for_models.data_path_train)
    df_val = pd.read_csv(paths_for_models.data_path_val)

    # Separare i dati in base a BLASTO NY per train
    blasto_train = df_train[df_train['BLASTO NY'] == 1]
    no_blasto_train = df_train[df_train['BLASTO NY'] == 0]

    # Separare i dati in base a BLASTO NY per validation
    blasto_val = df_val[df_val['BLASTO NY'] == 1]
    no_blasto_val = df_val[df_val['BLASTO NY'] == 0]

    # Creare i plot per train e validation
    create_plot(blasto_train, no_blasto_train, 'Media dei valori temporali - Train Set', f'mean_train_data_normALL{conf.seed}_{conf.temporalDataType}.jpg')
    create_plot(blasto_val, no_blasto_val, 'Media dei valori temporali - Validation Set', f'mean_val_data_normALL{conf.seed}_{conf.temporalDataType}.jpg')

    # Carico i dati di test e creo il grafico per test
    df_test = pd.read_csv(paths_for_models.test_path)

    blasto_test = df_test[df_test['BLASTO NY'] == 1]
    no_blasto_test = df_test[df_test['BLASTO NY'] == 0]

    create_plot(blasto_test, no_blasto_test, 'Media dei valori temporali - Test Set', f'mean_test_data_{conf.seed}_{conf.temporalDataType}.jpg')
    

    print(f"salvati dati relativi a normalizzazione su tutte le serie temporali con seed {conf.seed} e su dati di {conf.temporalDataType}")

