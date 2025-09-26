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

from config import Config_02c_splitAndNormalization as conf
from config import utils
from _02c_splitAndnormalization._01_split_normalization import load_data, normalize_data, save_data, stratified_split
from _02c_splitAndnormalization._02_visualization import (visualize_normalized_data_single_pt,
                                                  create_and_save_plots_mean_temp_data, 
                                                  create_and_save_stratified_plots_mean_temp_data)
from _utils_.dimReduction import compute_UMAP, compute_tSNE, compute_UMAP_with_plotly


def import_original_db_and_merge_data(data, original_db_path):
    # Merge PN data with original and normalized data
    df_db = pd.read_excel(original_db_path)[['slide_well']]

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


def main(days_to_consider=conf.days_to_consider,
         train_size=conf.train_size,
         seed=conf.seed,
         temporalDataType = conf.temporalDataType,
         csv_file_path = conf.csv_file_path,
         method_optical_flow = conf.method_optical_flow,
         embedding_type=conf.embedding_type, 
         original_db_path=conf.path_original_excel,
         save_normalization_example_single_pt=conf.save_normalization_example_single_pt, 
         mean_data_visualization=conf.mean_data_visualization,
         specific_patient_to_analyse=conf.specific_patient_to_analyse, 
         mean_data_visualization_stratified=conf.mean_data_visualization_stratified,
         inf_quantile=conf.inf_quantile, 
         sup_quantile=conf.sup_quantile,
         initial_frames_to_cut=conf.initial_frames_to_cut):
    
    for day in days_to_consider:
        # Carica i dati
        max_frames = utils.num_frames_by_days(day)
        data = load_data(csv_file_path=csv_file_path, initial_frames_to_cut=initial_frames_to_cut, max_frames=max_frames)

        # Split stratificato
        train_data, val_data, test_data = stratified_split(data, train_size=train_size, seed=seed)

        # Normalizza i dati
        train_data_norm, val_data_norm, test_data_norm = normalize_data(train_data, val_data, test_data, inf_quantile=inf_quantile, sup_quantile=sup_quantile)

        # Salva i dataset normalizzati
        base_path = conf.get_normalized_base_path(day)
        save_data(train_data=train_data_norm, val_data=val_data_norm, test_data=test_data_norm, output_base_path=base_path, days_to_consider=day)

        if embedding_type:
            # Visualizzo embedding di un subset a scelta --> ottengo il percorso, ad esempio, del train (numero maggiore di dati)
            train_path, val_path, test_path = conf.get_paths(day)
            output_path_base = os.path.join(current_dir, method_optical_flow, "dim_reduction_files")
            os.makedirs(output_path_base, exist_ok=True)
            max_frames = utils.num_frames_by_days(day)

            if embedding_type.lower()=="umap":
                print("Computing UMAP...")
                compute_UMAP_with_plotly(csv_path=train_path, days_to_consider=day, max_frames=max_frames, output_path_base=output_path_base)
            elif embedding_type.lower()=="tsne":
                print("Computing tSNE...")
                compute_tSNE(csv_path=train_path, days_to_consider=day, max_frames=max_frames, output_path_base=output_path_base)
            else:
                print("Please select a valid dimensionality reduction method. You can choose from: umap, tsne")

        if save_normalization_example_single_pt:
            train_data_merged = import_original_db_and_merge_data(data=train_data, original_db_path=original_db_path)
            train_data_norm_merged = import_original_db_and_merge_data(data=train_data_norm, original_db_path=original_db_path)
            visualize_normalized_data_single_pt(
                original_data=train_data_merged,  # Use merged data
                normalized_data=train_data_norm_merged,  # Use merged data
                output_base=os.path.join(current_dir, method_optical_flow),
                specific_patient_id=specific_patient_to_analyse if specific_patient_to_analyse is not 0 else None,
                shift_x=initial_frames_to_cut
            )
        
        if mean_data_visualization:
            create_and_save_plots_mean_temp_data(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            output_base=os.path.join(current_dir, method_optical_flow),
            seed=seed,
            temporal_data_type=temporalDataType,
            days_to_consider=day,
            shift_x=initial_frames_to_cut
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
                output_base=os.path.join(current_dir, method_optical_flow),
                seed=seed,
                temporal_data_type=temporalDataType,
                days_to_consider=day,
                shift_x = initial_frames_to_cut
                )


if __name__ == "__main__":
    start_time = time.time()
    main(days_to_consider=conf.days_to_consider, 
         train_size=conf.train_size, 
         seed=conf.seed, 
         temporalDataType = conf.temporalDataType, 
         csv_file_path = conf.csv_file_path, 
         embedding_type=conf.embedding_type, 
         original_db_path=conf.path_original_excel,
         save_normalization_example_single_pt=conf.save_normalization_example_single_pt, 
         mean_data_visualization=conf.mean_data_visualization,
         specific_patient_to_analyse=conf.specific_patient_to_analyse, 
         mean_data_visualization_stratified=conf.mean_data_visualization_stratified,
         inf_quantile=conf.inf_quantile, 
         sup_quantile=conf.sup_quantile,
         initial_frames_to_cut=conf.initial_frames_to_cut)
    print("Execution time: ", str(time.time()-start_time), "seconds")
