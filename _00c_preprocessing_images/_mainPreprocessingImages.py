import time
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



def main(input_excel_path):

    # Percorso della directory sorgente (con i video equatoriali)
    src_dir = "/home/phd2/Scrivania/CorsoData/ScopeData_equator"
    # Percorso della directory di destinazione
    dest_dir = "/home/phd2/Scrivania/CorsoData/blastocisti"
    _01_check_and_prepare_images.organize_videos_by_classification(src_dir, dest_dir, input_excel_path)


    # Percorso principale delle cartelle
    root_dir = dest_dir
    # Percorso per il file di log
    log_file_path = os.path.join(root_dir, "corrupted_images_log.txt")
    # Controlla le immagini
    _01_check_and_prepare_images.check_for_corrupted_images(root_dir, log_file_path)

    # Percorso come prima (è dove ci sono le immagini da fixare)
    image_path = dest_dir
    # Fix delle immagini
    _01_check_and_prepare_images.fix_images_in_directory(image_path)



if __name__ == '__main__':
    start_time = time.time()
    main(conf.path_original_excel)
    print("Excecution time: " + str(time.time() - start_time) + "seconds")





