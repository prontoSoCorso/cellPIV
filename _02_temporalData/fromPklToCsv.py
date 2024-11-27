## Il file che ottengo con il flusso ottico è un file pickle. In questo script voglio creare il csv con
# le serie temporali e gli identificativi dei video, facendo un taglio al settimo giorno e facendo il padding
# delle serie temporali minori di 7 giorni ##

import pickle
import pandas as pd
import os
import numpy as np
import sys

# Rileva il percorso della cartella genitore, che sarà la stessa in cui ho il file da convertire
current_dir = os.path.dirname(os.path.abspath(__file__))

# Individua la cartella 'cellPIV' come riferimento
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while os.path.basename(parent_dir) != "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import utils as conf

# Carica il file .pkl (Da cambiare in base al file che voglio convertire)
dict_in = "sum_mean_mag" + "_dict" + "_Farneback" + ".pkl"
csv_out = "sum_mean_mag" + "_Farneback" + ".csv"

with open(os.path.join(current_dir, dict_in), 'rb') as f:
    data = pickle.load(f)

# Processa i dati
processed_data = []  # Lista per raccogliere le righe

for key, value in data.items():
    if len(value) > conf.num_frames_7Days:
        # Tronca le serie troppo lunghe
        value = value[:conf.num_frames_7Days]
    elif len(value) < conf.num_frames_7Days:
        # Applica padding con zeri
        padding_length = conf.num_frames_7Days - len(value)
        value = np.pad(value, (0, padding_length), 'constant', constant_values=0)

    # Aggiungi una riga con chiave + valori
    processed_data.append([key] + list(value))

columns = ['dish_well'] + [f"time_{i}" for i in range(conf.num_frames_7Days)]  # Intestazioni
df = pd.DataFrame(processed_data, columns=columns)

# Salva il DataFrame in formato .csv
df.to_csv(os.path.join(parent_dir, '_02_temporalData', 'final_series', csv_out), index=False)
