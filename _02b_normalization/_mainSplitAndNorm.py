import os
import sys
import time
import pandas as pd

# Rileva il percorso della cartella genitore, che sarà la stessa in cui ho il file da convertire
current_dir = os.path.dirname(os.path.abspath(__file__))

# Individua la cartella 'cellPIV' come riferimento
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while os.path.basename(parent_dir) != "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_02b_normalization as conf
from config import utils
from _02b_normalization._01_split_normalization import load_data, normalize_data, save_data, stratified_split
from _02b_normalization._02_visualization import visualize_normalized_data_single_pt, create_and_save_plots_mean_temp_data, create_and_save_stratified_plots_mean_temp_data
from _utils_.dimReduction import compute_UMAP, compute_tSNE


def import_original_db_and_merge_data(data, original_db_path=os.path.join(parent_dir, "DB morpheus UniPV.xlsx")):
    # Merge PN data with original and normalized data
    db_file = original_db_path
    df_db = pd.read_excel(db_file)[['slide_well', 'PN']]

    def merge_pn_data(df):
            merged = pd.merge(df, df_db, 
                              left_on='dish_well', 
                              right_on='slide_well', 
                              how='left')
            merged['merged_PN'] = merged['PN'].astype(str)
            return merged.drop(columns=['slide_well'])
    
    # Merge with original and normalized data
    data_merged = merge_pn_data(data)

    return data_merged




def main(days_to_consider=conf.days_to_consider, train_size=conf.train_size, seed=conf.seed, embedding_type=conf.embedding_type, original_db_path=os.path.join(parent_dir, "DB morpheus UniPV.xlsx"),
         save_normalization_example_single_pt=True, mean_data_visualization=True, specific_patient_to_analyse=None, mean_data_visualization_stratified = True,
         temporalDataType = conf.temporalDataType, csv_file_path = conf.csv_file_path):
    
    # Carica i dati
    max_frames = utils.num_frames_by_days(days_to_consider)
    data = load_data(csv_file_path=csv_file_path, max_frames=max_frames)

    # Split stratificato
    train_data, val_data, test_data = stratified_split(data, train_size=train_size, seed=seed)

    # Normalizza i dati
    train_data_norm, val_data_norm, test_data_norm = normalize_data(train_data, val_data, test_data)

    # Salva i dataset normalizzati
    base_path = conf.get_normalized_base_path(days_to_consider)
    save_data(train_data=train_data_norm, val_data=val_data_norm, test_data=test_data_norm, output_base_path=base_path, days_to_consider=days_to_consider)

    if embedding_type:
        # Visualizzo embedding di un subset a scelta --> ottengo il percorso, ad esempio, del train (numero maggiore di dati)
        train_path, val_path, test_path = conf.get_paths(days_to_consider)
        output_path_base = os.path.join(current_dir, "dim_reduction_files")
        os.makedirs(output_path_base, exist_ok=True)
        max_frames = utils.num_frames_by_days(days_to_consider)

        if embedding_type.lower()=="umap":
            print("Computing UMAP...")
            compute_UMAP(csv_path=train_path, days_to_consider=days_to_consider, max_frames=max_frames, output_path_base=output_path_base)
        elif embedding_type.lower()=="tsne":
            print("Computing tSNE...")
            compute_tSNE(csv_path=train_path, days_to_consider=days_to_consider, max_frames=max_frames, output_path_base=output_path_base)
        else:
            print("Please select a valid dimensionality reduction method. You can choose from: umap, tsne")

    if save_normalization_example_single_pt:
        train_data_merged = import_original_db_and_merge_data(data=train_data, original_db_path=original_db_path)
        train_data_norm_merged = import_original_db_and_merge_data(data=train_data_norm, original_db_path=original_db_path)
        visualize_normalized_data_single_pt(
            original_data=train_data_merged,  # Use merged data
            normalized_data=train_data_norm_merged,  # Use merged data
            output_base=current_dir,
            specific_patient_id=specific_patient_to_analyse
        )
    
    if mean_data_visualization:
        create_and_save_plots_mean_temp_data(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        output_base=current_dir,
        seed=seed,
        temporal_data_type=temporalDataType,
        days_to_consider=days_to_consider
        )

    if mean_data_visualization_stratified:
        train_merged = import_original_db_and_merge_data(data=train_data, original_db_path=original_db_path)
        val_merged = import_original_db_and_merge_data(data=val_data, original_db_path=original_db_path)
        test_merged = import_original_db_and_merge_data(data=test_data, original_db_path=original_db_path)

        # Create stratified plots
        create_and_save_stratified_plots_mean_temp_data(
            train_merged=train_merged,
            val_merged=val_merged,
            test_merged=test_merged,
            output_base=current_dir,
            seed=seed,
            temporal_data_type=temporalDataType,
            days_to_consider=days_to_consider
            )


if __name__ == "__main__":
    start_time = time.time()
    main(days_to_consider=3, train_size=0.7, seed=42, temporalDataType = conf.temporalDataType, csv_file_path = conf.csv_file_path, 
         embedding_type="", save_normalization_example_single_pt=False, mean_data_visualization=True,
         specific_patient_to_analyse=61, mean_data_visualization_stratified=True)
    print("Execution time: ", str(time.time()-start_time), "seconds")
