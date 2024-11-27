import numpy as np
import pandas as pd
from sktime.datatypes._panel._convert import from_3d_numpy_to_nested
from sktime.classification.hybrid import HIVECOTEV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dati di esempio per un singolo video (num_channels = 3, num_frames = 10)
series_data = np.array([[
    [0.5, 0.6, 0.7, 0.5, 0.4, 0.6, 0.7, 0.5, 0.6, 0.7],  # sum mean mag
    [0.1, 0.2, 0.3, 0.2, 0.1, 0.3, 0.4, 0.2, 0.3, 0.4],  # vorticity
    [0.8, 0.9, 0.85, 0.8, 0.75, 0.85, 0.9, 0.8, 0.85, 0.9]  # area della cellula
]])

# Eventi di esempio per un singolo video (num_video = 1, num_events = 7)
events_data = np.array([
    [2, 4, 7, -1, -1, 1, 9]  # t2, t3, t4, t6, t8, tPN, tPNF
])

# Converti le serie temporali in formato sktime
series_df = from_3d_numpy_to_nested(series_data)

# Creare un MultiIndex per le serie temporali
series_df.index = pd.MultiIndex.from_product([range(series_df.shape[0]), ['series']], names=["sample", "type"])
series_df.columns = ['sum_mean_mag', 'vorticity', 'area']
print("Series DataFrame:\n", series_df.head())

# Creare un DataFrame per gli eventi
events_df = pd.DataFrame(events_data, columns=['t2', 't3', 't4', 't6', 't8', 'tPN', 'tPNF'])

# Creare un MultiIndex per gli eventi
events_df.index = pd.MultiIndex.from_product([range(events_df.shape[0]), ['events']], names=["sample", "type"])
print("Events DataFrame:\n", events_df.head())

# Combina le serie temporali con gli eventi
combined_df = pd.concat([series_df, events_df], axis=0).sort_index(level=0)
print("Combined DataFrame:\n", combined_df.head())

# Esempio di etichetta binaria per un video
labels = np.array([1])  # Ad esempio, 1 rappresenta una condizione positiva, 0 negativa

# Inizializza e allena HIVE-COTE 2.0
hivecote = HIVECOTEV2(random_state=42)
hivecote.fit(combined_df, labels)

# Predici sui dati (per esempio sui dati di training stesso in questo esempio)
pred = hivecote.predict(combined_df)

# Valuta le prestazioni (ad esempio, con accuratezza)
accuracy = accuracy_score(labels, pred)

print(f"Accuracy: {accuracy:.2f}")
