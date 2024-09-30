import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


# Definisci i percorsi dei file
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_02_temporalData as conf

df_original = pd.read_csv(conf.output_csv_file_path)
df_norm = pd.read_csv(conf.output_csvNormalized_file_path)


# Plot dati normalizzati
patient_id_example = 55

# Estrai i dati del paziente specificato prima della normalizzazione
original_signals = df_original[df_original['patient_id'] == patient_id_example]

# Estrai i dati del paziente specificato dopo la normalizzazione
normalized_signals = df_norm[df_norm['patient_id'] == patient_id_example]

# Plot dei segnali originali
plt.figure(figsize=(14, 6))
plt.subplot(2,1,1)
for row in range(original_signals.shape[0]):
    plt.plot(original_signals.iloc[row, 3:], alpha=0.5, label=original_signals.iloc[row,1])
plt.title(f'Segnali originali per il paziente {patient_id_example}')
plt.xlabel('Time Point')
plt.xticks(range(0, 290, 20))  # Show labels every 10th time point
plt.ylabel('Value')
plt.legend()

plt.subplot(2,1,2)
for row in range(normalized_signals.shape[0]):
    plt.plot(normalized_signals.iloc[row, 3:], alpha=0.5, label=normalized_signals.iloc[row,1])
plt.title(f'Segnali normalizzati per il paziente {patient_id_example}')
plt.xlabel('Time Point')
plt.xticks(range(0, 290, 20))  # Show labels every 10th time point
plt.ylabel('Value')
plt.legend()
plt.show()