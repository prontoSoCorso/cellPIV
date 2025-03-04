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






def main():
    model_to_train = "ROCKET"   # possible choices: ["ROCKET","LSTMFCN","ConvTran"]
    













if __name__ == "__main__":
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    print("")



