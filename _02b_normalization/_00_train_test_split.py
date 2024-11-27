## Prima split per anno, ora a partire da un singolo file csv ne creo tre, uno per train, uno per validation ed uno per test 
# Importante: i vari well del singolo paziente non possono essere in subset differenti! ##

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





















