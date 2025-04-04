import os
import sys
# Configurazione dei percorsi e dei parametri
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_00_preprocessing as conf
from _01_extract_images import extract_frames
from _02_extract_equatore import copy_equatorial_frames

def count_folders(path_main_folder):
    # Contatore per le sottocartelle
    total_subfolders = 0

    # Itera su ogni cartella nella directory principale
    for folder in os.listdir(path_main_folder):
        folder_path = os.path.join(path_main_folder, folder)
        
        # Controlla se è una directory
        if os.path.isdir(folder_path):
            # Conta le sottocartelle all'interno della cartella
            subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
            total_subfolders += len(subfolders)

    print(f"Numero totale di sottocartelle: {total_subfolders}")


def main(extract_pdb=True, extract_equator=True, count_final_folders=True, 
         input_dir=conf.input_dir_pdb_files, output_dir_extracted_pdb_files=conf.output_dir_extracted_pdb_files, log_file=conf.log_file_pdb_extraction, 
         src_dir=conf.src_dir_extracted_pdb, dest_dir=conf.dest_dir_extracted_equator,
         path_main_folder=conf.path_main_folder):
    ##############################
    # Extraction from pdb files
    ##############################
    if extract_pdb:
        extract_frames(input_dir=input_dir, output_dir=output_dir_extracted_pdb_files, log_file=log_file)

    ##############################
    # Selecting equatorial images
    ##############################
    if extract_equator:
        copy_equatorial_frames(src_dir=src_dir, dest_dir=dest_dir)

    ##############################
    # Count final folders
    ##############################
    if count_final_folders:
        count_folders(path_main_folder=path_main_folder)


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    print("Execution time: ", str(time.time()-start_time), "seconds")
