import os
import sys
# Configurazione dei percorsi e dei parametri
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path) 
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_00_preprocessing as conf
from _01_extract_images import extract_frames
from _02_extract_equatore import copy_equatorial_frames
from _03_check_empty_subfolders import check_empty_subfolders
from _04_stats_timing import main as compute_stats_timing
from _05_copy_and_rename_with_hpi import main as copy_and_rename_hpi

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


def main(extract_pdb=False, extract_equator=False, count_final_folders=False, check_empty=False, stats_timing=True, copy_and_rename=False,
         input_dir=conf.input_dir_pdb_files, output_dir_extracted_pdb_files=conf.output_dir_extracted_pdb_files, log_extraction_file=conf.log_file_pdb_extraction, 
         src_dir=conf.src_dir_extracted_pdb, dest_dir=conf.dest_dir_extracted_equator,
         log_empty_file = None,
         scope_eq_dir=conf.src_dir_extracted_equator, scope_final_dir=conf.dest_dir_time_conversion, log_stats_file=None, 
         path_main_folder=conf.path_main_folder):
    ##############################
    # Extraction from pdb files
    ##############################
    if extract_pdb:
        extract_frames(input_dir=input_dir, output_dir=output_dir_extracted_pdb_files, log_file=log_extraction_file,
                       first_year=2019, last_year=2020)

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


    ##############################
    # Check empty folders
    ##############################
    if check_empty:
        check_empty_subfolders(input_dir=path_main_folder, log_dir=log_empty_file)

    ##############################
    # Stats
    ##############################
    if stats_timing:
        compute_stats_timing(input_dir=scope_eq_dir, output_dir=scope_final_dir, log_file=log_stats_file)

    ##############################
    # Copy all eligible videos and rename each frame with the time conversion in hours post insemination
    ##############################
    if copy_and_rename:
        copy_and_rename_hpi(input_dir=scope_eq_dir, output_dir=scope_final_dir)


if __name__ == "__main__":
    import time
    start_time = time.time()
    log_dir = "logs_and_plots"
    main(log_extraction_file=os.path.join(current_dir, log_dir, "extraction_log.txt"),
         log_stats_file=os.path.join(current_dir, log_dir, "stats_timing_log.txt"),
         log_empty_file=os.path.join(current_dir, log_dir, "empty_folders_report.txt"),)
    print("Execution time: ", str(time.time()-start_time), "seconds")
