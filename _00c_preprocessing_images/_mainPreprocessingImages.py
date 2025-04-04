import os
import sys
# Configurazione dei percorsi e dei parametri
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_00_preprocessing as conf
from _00c_preprocessing_images import _01_check_and_prepare_images


def main(src_dir=conf.path_main_folder, dest_dir=conf.dest_dir_blastoData, input_excel_path=conf.path_original_excel):
    
    # Dividing videos in two folder (blasto and no_blasto)
    _01_check_and_prepare_images.organize_videos_by_classification(src_dir, dest_dir, input_excel_path)

    # Percorso per il file di log
    log_file_path = os.path.join(dest_dir, "corrupted_images_log.txt")
    # Controlla le immagini
    _01_check_and_prepare_images.check_for_corrupted_images(dest_dir, log_file_path)

    # Fix delle immagini --> Percorso come prima (è dove ci sono le immagini da fixare)
    _01_check_and_prepare_images.fix_images_in_directory(dest_dir)


if __name__ == '__main__':
    import time
    start_time = time.time()
    main(src_dir=conf.path_main_folder, dest_dir=conf.dest_dir_blastoData, input_excel_path=conf.path_original_excel)
    print("Excecution time: " + str(time.time() - start_time) + "seconds")