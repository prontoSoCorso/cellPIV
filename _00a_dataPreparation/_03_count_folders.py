import os

# Percorso della directory principale
path = "/home/phd2/Scrivania/CorsoData/ScopeData_equator"

# Contatore per le sottocartelle
total_subfolders = 0

# Itera su ogni cartella nella directory principale
for folder in os.listdir(path):
    folder_path = os.path.join(path, folder)
    
    # Controlla se è una directory
    if os.path.isdir(folder_path):
        # Conta le sottocartelle all'interno della cartella
        subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        total_subfolders += len(subfolders)

print(f"Numero totale di sottocartelle: {total_subfolders}")