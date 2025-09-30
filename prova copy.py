
# Importa "/home/phd2/Scrivania/CorsoRepo/cellPIV/datasets/Farneback/subsets/Normalized_sum_mean_mag_5Days_train.csv"
# e verifica se ci sono NaN o infiniti
import os
import pandas as pd
import sys
import numpy as np

df = pd.read_csv("/home/phd2/Scrivania/CorsoRepo/cellPIV/datasets/Farneback/subsets/Normalized_sum_mean_mag_5Days_test.csv")

# select only the "hours" columns
print(df.head())

df_hours = df[[c for c in df.columns if c.endswith('h')]]
print(df_hours.head())

# Verifica NaN
if df_hours.isnull().values.any():
    count_NaN = df_hours.isnull().sum().sum()
    print(f"Il DataFrame contiene {count_NaN} valori NaN.")

# Verifica infiniti
if np.isinf(df_hours.values).any():
    print("Il DataFrame contiene valori infiniti.")
