import os
import pickle
import sys
import time
from process_optical_flow import process_frames

# Configurazione dei percorsi e dei parametri
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_01_OpticalFlow as conf
from config import user_paths as myPaths
from config import utils as utils

def main():
    # Success/Error counters
    n_video = 0
    n_video_blasto = 0
    n_video_no_blasto = 0
    n_video_error_blasto = 0
    n_video_error_no_blasto = 0
    n_video_success_blasto = 0
    n_video_success_no_blasto = 0
    errors_blasto = []
    errors_no_blasto = []

    # Dizionari per memorizzare le metriche
    mean_magnitude_dict = {}
    vorticity_dict = {}
    hybrid_dict = {}
    sum_mean_mag_dict = {}

    print(f"\n===== Metodo utilizzato per il calcolo del flusso ottico: {conf.method_optical_flow} =====\n")

    for class_sample in ['blasto', 'no_blasto']:
        path_all_folders = os.path.join(myPaths.path_BlastoData, class_sample)
        total_videos = len(os.listdir(path_all_folders))

        for idx, sample in enumerate(os.listdir(path_all_folders), start=1):
            n_video += 1
            if class_sample == "blasto":
                n_video_blasto += 1
            else:
                n_video_no_blasto += 1

            print(f"Calcolando il flusso ottico del video {idx}/{total_videos} ({class_sample})")

            try:
                sample_path = os.path.join(path_all_folders, sample)
                mean_magnitude, vorticity, hybrid, sum_mean_mag = process_frames(
                    sample_path, sample, 
                    utils.img_size, conf.num_minimum_frames, conf.num_initial_frames_to_cut, 
                    conf.num_forward_frame, conf.method_optical_flow
                )
                
                mean_magnitude_dict[sample] = mean_magnitude
                vorticity_dict[sample] = vorticity
                hybrid_dict[sample] = hybrid
                sum_mean_mag_dict[sample] = sum_mean_mag

                if class_sample == "blasto":
                    n_video_success_blasto += 1
                else:
                    n_video_success_no_blasto += 1

            except Exception as e:
                if class_sample == "blasto":
                    n_video_error_blasto += 1
                    errors_blasto.append(sample)
                else:
                    n_video_error_no_blasto += 1
                    errors_no_blasto.append(sample)

                print('-------------------')
                print(f"Errore nel sample: {sample}")
                print(e)
                print('-------------------')
                continue

    print('===================')
    print("Terminata Elaborazione...")
    print('===================')

    # Stampo quanti frame con successo e quanti errori
    print('===================')
    print('===================')
    print(f"Analizzati {n_video} video: {n_video_blasto} blasto, {n_video_no_blasto} no_blasto")
    print(f"Successi: {n_video_success_blasto} blasto, {n_video_success_no_blasto} no_blasto")
    print(f"Errori: {n_video_error_blasto} blasto, {n_video_error_no_blasto} no_blasto")

    print('===================')
    print('===================')
    print(f'Errors in "blasto":', errors_blasto)

    print('===================')
    print('===================')
    print(f'Errors in "no_blasto":', errors_no_blasto)

    # Salvataggio dei risultati
    temporal_data_directory = os.path.join(parent_dir, '_02_temporalData')
    for dict_name, dict_data in zip(
        ['mean_magnitude_dict', 'vorticity_dict', 'hybrid_dict', 'sum_mean_mag_dict'],
        [mean_magnitude_dict, vorticity_dict, hybrid_dict, sum_mean_mag_dict]):
        file_path = os.path.join(temporal_data_directory, f"{dict_name}_{conf.method_optical_flow}.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(dict_data, f)

if __name__ == "__main__":
    start_time = time.time()
    # execution_time = timeit.timeit(main, number=1)
    main()
    print("Execution time:", str(time.time()-start_time), "seconds")
