import os
import sys
import time


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
from _02_temporalData._02_dimReduction import compute_tSNE, compute_UMAP


def main(embedding_type="UMAP", num_max_days=7, days_to_consider_dim_reduction=7):
    # importo e trasformo i file pkl salvati al passo precedente (o qualsiasi file pickle con stesso formato: key (dish_well) e valori della serie)
    os.makedirs(conf.temporal_csv_path, exist_ok=True)
    path_pkl = conf.path_pkl
    output_temporal_csv_path = conf.temporal_csv_path
    num_frames_MaxDays = utils.num_frames_by_days(num_max_days)
    fromPickleToCsv(current_dir=current_dir, path_pkl=path_pkl, output_temporal_csv_path=output_temporal_csv_path, num_frames_MaxDays=num_frames_MaxDays)
    
    # creo il csv finale con i metadati ed i values della sum_mean_mag (metrica scelta perché più robusta e affidabile)
    input_temporal_data_path = conf.temporal_csv_path
    original_csv_path = conf.csv_file_Danilo_path
    output_final_csv = conf.final_csv_path
    create_final_csv(input_temporal_csv_path=input_temporal_data_path, original_csv_path=original_csv_path, output_final_csv_path=output_final_csv)

    # Rappresentazione visuale dei dati: embedding tramite umap e/o tSNE
    final_csv_path = conf.final_csv_path
    output_path_base = conf.csv_file_Danilo_path

    for day in days_to_consider_dim_reduction:
        max_frames = utils.num_frames_by_days(days_to_consider_dim_reduction)

        if embedding_type.lower()=="umap":
            print("Computing UMAP...")
            compute_UMAP(final_csv_path=final_csv_path, days_to_consider=day, max_frames=max_frames, output_path_base=output_path_base)
        elif embedding_type.lower()=="tsne":
            print("Computing tSNE...")
            compute_tSNE(final_csv_path=final_csv_path, days_to_consider=day, max_frames=max_frames, output_path_base=output_path_base)
        else:
            print("Please select a valid dimensionality reduction method. You can choose from: umap, tsne")


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("Execution time: ", str(time.time()-start_time), "seconds")



