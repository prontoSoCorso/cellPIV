import time
import os

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



def main(extract_pdb=True, extract_equator=True, count_final_folders=True):
    ##############################
    # Extraction from pdb files
    ##############################
    if extract_pdb:
        input_dir = "/home/phd2/Scrivania/CorsoData/ScopeData"
        output_dir = "/home/phd2/Scrivania/CorsoData/ScopeData_extracted"
        log_file = "/home/phd2/Scrivania/CorsoData/estrazione_log.txt"
        extract_frames(input_dir=input_dir, output_dir=output_dir, log_file=log_file)

    ##############################
    # Selecting equatorial images
    ##############################
    if extract_equator:
        src_dir = "/home/phd2/Scrivania/CorsoData/ScopeData_extracted"
        dest_dir = "/home/phd2/Scrivania/CorsoData/ScopeData_equator"
        copy_equatorial_frames(src_dir=src_dir, dest_dir=dest_dir)

    ##############################
    # Count final folders
    ##############################
    if count_final_folders:
        path_main_folder = "/home/phd2/Scrivania/CorsoData/ScopeData_equator"
        count_folders(path_main_folder=path_main_folder)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Execution time: ", str(time.time()-start_time), "seconds")
