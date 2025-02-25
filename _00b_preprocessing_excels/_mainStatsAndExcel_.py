import pandas as pd
import os
import time

# Ottieni il percorso del file corrente e cartella
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

# Risali la gerarchia fino alla cartella "cellPIV"
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)

import sys
sys.path.append(parent_dir)

from config import Config_00_preprocessing as conf
from _00b_preprocessing_excels import _01_prepareExcelAndAddID, _02_calculateAndPlotStatistics, _03_checkAgeSpermCulture
from _utils_ import plot_and_save_stratified_distribution


def main(path_original_excel, path_added_ID, path_double_dish_excel):
    # Rendo colonne conformi a come devo elaborare dopo
    blasto_labels_df = _01_prepareExcelAndAddID.merge_sheets_and_prepare_csv(path_original_excel)
    
    # Aggiungo ID del paziente in modo da avere informazione per successivo split o se voglio altre statistiche
    _01_prepareExcelAndAddID.match_double_dishes_and_create_csv(blasto_labels_df=blasto_labels_df, output_path_added_ID=path_added_ID, path_double_dish_excel=path_double_dish_excel)

    # Faccio statistiche su blasto per anno, total samples etc.
    _02_calculateAndPlotStatistics.calculate_and_plot_statistics(input_csv_path=path_added_ID, output_dir=current_dir)

    # Verifico che le variabili non siano correlate in modo diretto e univariato con l'outcome
    _03_checkAgeSpermCulture.check_variable_independence(input_csv_path=path_added_ID, output_dir=current_dir)

    # Calcolo e plot delle statistiche stratificate per tipologia di PN
    output_path = os.path.join(current_dir, "plots_pn_statistics.png")
    plot_and_save_stratified_distribution.main(input_csv_path=path_added_ID, output_path=output_path)


if __name__ == '__main__':
    start_time = time.time()

    main(path_original_excel=conf.path_original_excel, path_added_ID=conf.path_addedID_csv, path_double_dish_excel=conf.path_double_dish_excel)

    print("Excecution time: " + str(time.time() - start_time) + "seconds")










