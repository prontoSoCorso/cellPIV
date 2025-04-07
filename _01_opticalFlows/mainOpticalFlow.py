import os
import pickle
import sys
import time
import logging
import numpy as np
from _01_opticalFlows._process_optical_flow import process_frames

# Configurazione dei percorsi e dei parametri
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = current_dir
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_01_OpticalFlow as conf
from config import user_paths as myPaths
from config import utils as utils

def main(method_optical_flow=conf.method_optical_flow, path_BlastoData=myPaths.path_BlastoData, 
         img_size=conf.img_size, num_minimum_frames=conf.num_minimum_frames, 
         num_initial_frames_to_cut=conf.num_initial_frames_to_cut, num_forward_frame=conf.num_forward_frame,
         output_metrics_base_dir = os.path.join(current_dir, "metrics_examples"),
         save_metrics=conf.save_metrics,
         output_path_optical_flow_images=conf.output_path_optical_flow_images,
         save_overlay_optical_flow=conf.save_overlay_optical_flow,
         save_final_data=conf.save_final_data):
    
    # Specify the desired log file path
    logging.basicConfig(filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                              f'optical_flow_complete_analysis_{method_optical_flow}.log'), 
                        level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
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

    logging.info(f"Starting analysis with method: {method_optical_flow}")

    for class_sample in ['blasto', 'no_blasto']:
        path_all_folders = os.path.join(path_BlastoData, class_sample)
        total_videos = len(os.listdir(path_all_folders))

        for idx, sample in enumerate(os.listdir(path_all_folders), start=1):
            n_video += 1
            if class_sample == "blasto":
                n_video_blasto += 1
            else:
                n_video_no_blasto += 1

            logging.info(f"Processing video n: {idx}/{total_videos}, {class_sample}/{sample}")

            try:
                sample_path = os.path.join(path_all_folders, sample)
                output_metrics_base_path=os.path.join(output_metrics_base_dir, method_optical_flow, class_sample)
                os.makedirs(output_metrics_base_path, exist_ok=True)
                output_path_images = os.path.join(output_path_optical_flow_images, method_optical_flow, class_sample, sample)
                os.makedirs(output_path_images, exist_ok=True)

                metrics = process_frames(
                    folder_path=sample_path, 
                    dish_well=sample, 
                    img_size=img_size, 
                    num_minimum_frames=num_minimum_frames, 
                    num_initial_frames_to_cut=num_initial_frames_to_cut, 
                    num_forward_frame=num_forward_frame, 
                    method_optical_flow=method_optical_flow,
                    output_metrics_base_path=output_metrics_base_path,
                    save_metrics=save_metrics,
                    save_overlay_optical_flow=save_overlay_optical_flow,
                    output_path_images_with_optical_flow=output_path_images
                    )
                
                mean_magnitude_dict[sample] = np.array(metrics["mean_magnitude"]).astype(float)
                vorticity_dict[sample] = np.array(metrics["vorticity"]).astype(float)
                hybrid_dict[sample] = np.array(metrics["hybrid"]).astype(float)
                sum_mean_mag_dict[sample] = np.array(metrics["sum_mean_mag"]).astype(float)

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
    if save_final_data:
        temporal_data_directory = os.path.join(parent_dir, '_02_temporalData', f"files_all_days_{method_optical_flow}")
        os.makedirs(temporal_data_directory, exist_ok=True)
        for dict_name, dict_data in zip(
            ['mean_magnitude_dict', 'vorticity_dict', 'hybrid_dict', 'sum_mean_mag_dict'],
            [mean_magnitude_dict, vorticity_dict, hybrid_dict, sum_mean_mag_dict]):
            file_path = os.path.join(temporal_data_directory, f"{dict_name}_{method_optical_flow}.pkl")
            with open(file_path, 'wb') as f:
                pickle.dump(dict_data, f)

if __name__ == "__main__":
    start_time = time.time()
    # execution_time = timeit.timeit(main, number=1)
    main(method_optical_flow=conf.method_optical_flow, path_BlastoData=myPaths.path_BlastoData, 
         img_size=conf.img_size, num_minimum_frames=conf.num_minimum_frames, 
         num_initial_frames_to_cut=conf.num_initial_frames_to_cut, num_forward_frame=conf.num_forward_frame,
         output_metrics_base_dir = os.path.join(current_dir, "metrics_examples"),
         save_metrics=conf.save_metrics,
         output_path_optical_flow_images=conf.output_path_optical_flow_images,
         save_overlay_optical_flow=conf.save_overlay_optical_flow)
    print("Execution time:", str(time.time()-start_time), "seconds")
