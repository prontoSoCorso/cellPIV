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
from _utils_.dimReduction import compute_tSNE, compute_UMAP


def main(embedding_type=conf.embedding_type, num_max_days=conf.num_max_days, path_pkl = conf.path_pkl,
         days_to_consider_for_dim_reduction=conf.days_to_consider_for_dim_reduction, temporal_csv_path = conf.temporal_csv_path,
         original_csv_path = conf.csv_file_Danilo_path, final_csv_path = conf.final_csv_path):
    
    # Dopo aver importato i file pkl salvati al passo precedente, li trasformo 
    # (o qualsiasi file pickle con stesso formato: key (dish_well) e valori della serie)
    num_frames_MaxDays = utils.num_frames_by_days(num_max_days),    
    fromPickleToCsv(current_dir=current_dir, path_pkl=path_pkl, output_temporal_csv_path=temporal_csv_path, num_frames_MaxDays=num_frames_MaxDays)
    
    # creo il csv finale con i metadati ed i values della sum_mean_mag (metrica scelta perché più robusta e affidabile)
    create_final_csv(input_temporal_csv_path=temporal_csv_path, original_csv_path=original_csv_path, output_final_csv_path=final_csv_path)

    # Rappresentazione visuale dei dati: embedding tramite umap e/o tSNE
    output_path_base = os.path.join(current_dir, "dim_reduction_files")
    os.makedirs(output_path_base, exist_ok=True)

    if embedding_type:
        for day in days_to_consider_for_dim_reduction:
            max_frames = utils.num_frames_by_days(day)

            if embedding_type.lower()=="umap":
                print("Computing UMAP...")
                compute_UMAP(csv_path=final_csv_path, days_to_consider=day, max_frames=max_frames, output_path_base=output_path_base)
            elif embedding_type.lower()=="tsne":
                print("Computing tSNE...")
                compute_tSNE(csv_path=final_csv_path, days_to_consider=day, max_frames=max_frames, output_path_base=output_path_base)
            else:
                print("Please select a valid dimensionality reduction method. You can choose from: umap, tsne")


if __name__ == '__main__':
    start_time = time.time()
    main(embedding_type="UMAP", num_max_days=7, days_to_consider_for_dim_reduction=[1,3,5,7],
         path_pkl = conf.path_pkl, temporal_csv_path = conf.temporal_csv_path,
         original_csv_path = conf.csv_file_Danilo_path, final_csv_path = conf.final_csv_path)
    print("Execution time: ", str(time.time()-start_time), "seconds")
