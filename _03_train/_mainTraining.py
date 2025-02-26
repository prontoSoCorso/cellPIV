import sys
import os
import pandas as pd
import timeit
import seaborn as sns
import matplotlib.pyplot as plt
import torch

# Configurazione dei percorsi
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_03_train as conf


if __name__ == "__main__":
    execution_time = timeit.timeit(lambda: main(
        train_path="path/to/train.csv",
        val_path="path/to/val.csv",
        test_path="path/to/test.csv",
        kernels=[1000, 5000, 10000],
        seed=42,
        output_dir="./models/rocket/",
        days_to_consider=7
    ), number=1)
    print(f"Execution time: {execution_time:.2f} seconds")
