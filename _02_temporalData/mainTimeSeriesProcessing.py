import os
import sys
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Rileva il percorso della cartella genitore, che sarà la stessa in cui ho il file da convertire
current_dir = os.path.dirname(os.path.abspath(__file__))

# Individua la cartella 'cellPIV' come riferimento
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while os.path.basename(parent_dir) != "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_02_temporalData as conf
from config import utils
from _02_temporalData._01_fromPklToCsv import fromPickleToCsv, create_final_csv
from _utils_.dimReduction import compute_tSNE, compute_UMAP, compute_UMAP_with_plotly


def main(convert_pkl_to_csv = conf.convert_pkl_to_csv,
         path_pkl = os.path.join(current_dir,conf.path_pkl),
         temporal_csv_path = conf.temporal_csv_path,
         original_csv_path = conf.csv_file_Danilo_path,
         final_csv_path = conf.opt_flow_csv_path,
         path_output_dim_reduction_files=os.path.join(current_dir, conf.path_output_dim_reduction_files),
         embedding_type=conf.embedding_type,
         use_plotly_lib=conf.use_plotly_lib,
         num_max_days=conf.num_max_days,
         days_to_consider_for_dim_reduction=conf.days_to_consider_for_dim_reduction
         ):
    
    if convert_pkl_to_csv:
        # Import pkl files and transformation into csv --> key (dish_well), values (temporal values))
        num_frames_MaxDays = utils.num_frames_by_days(num_max_days)    
        fromPickleToCsv(path_pkl=path_pkl, 
                        output_temporal_csv_path=temporal_csv_path, 
                        num_frames_MaxDays=num_frames_MaxDays)
        
        # creo il csv finale con i metadati ed i values della sum_mean_mag (metrica scelta perché più robusta e affidabile)
        create_final_csv(input_temporal_csv_path=temporal_csv_path, original_csv_path=original_csv_path, output_final_csv_path=final_csv_path)

    # Rappresentazione visuale dei dati: embedding tramite umap e/o tSNE
    if embedding_type:
        os.makedirs(path_output_dim_reduction_files, exist_ok=True)
        for day in days_to_consider_for_dim_reduction:
            max_frames = utils.num_frames_by_days(day)

            if embedding_type.lower()=="umap":
                if use_plotly_lib:
                    print(f"Computing UMAP {day}Days with plotly...")
                    compute_UMAP_with_plotly(csv_path=final_csv_path, days_to_consider=day, max_frames=max_frames, output_path_base=path_output_dim_reduction_files)
                else:
                    print(f"Computing UMAP {day}Days...")
                    compute_UMAP(csv_path=final_csv_path, days_to_consider=day, max_frames=max_frames, output_path_base=path_output_dim_reduction_files)
            elif embedding_type.lower()=="tsne":
                print(f"Computing tSNE {day}Days...")
                compute_tSNE(csv_path=final_csv_path, days_to_consider=day, max_frames=max_frames, output_path_base=path_output_dim_reduction_files)
            else:
                print("Please select a valid dimensionality reduction method. You can choose from: umap, tsne")


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("Execution time: ", str(time.time()-start_time), "seconds")