import os
import pickle
import cv2
import numpy as np
import timeit
from myFunctions import calculate_vorticity, sort_files_by_slice_number

import sys
# Configurazione dei percorsi e dei parametri
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_01_OpticalFlow_Farneback as conf
from config import user_paths as myPaths
from config import utils as utils

class InsufficientFramesError(Exception):
    """Eccezione sollevata quando il numero di frame è insufficiente."""
    pass

class InvalidImageSizeError(Exception):
    """Eccezione sollevata quando un'immagine non ha la dimensione 500x500."""
    pass

def compute_optical_flowFarneback(prev_frame, current_frame):
    # Parametri per l'algoritmo Farneback
    fb_params = dict(pyr_scale=conf.pyr_scale,
                     levels=conf.levels,
                     winsize=conf.winSize,
                     iterations=conf.iterations,
                     poly_n=conf.poly_n,
                     poly_sigma=conf.poly_sigma,
                     flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    # Calcolo il flusso ottico tra i frame
    flow = cv2.calcOpticalFlowFarneback(prev_frame, current_frame, None, **fb_params)

    # Calcola la magnitudo e l'angolo del flusso ottico
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Converti l'angolo in gradi
    angle_degrees = np.rad2deg(angle) % 360

    return magnitude, angle_degrees, flow

def process_frames(folder_path, dish_well):
    target_size = (utils.img_size, utils.img_size)    # Impongo dimensione immagini

    # Get the list of frames in the folder
    frame_files = sort_files_by_slice_number(os.listdir(folder_path))

    # Controllo se il numero di frame è sufficiente
    if len(frame_files) < 300:
        raise InsufficientFramesError(f"Il numero di frame nel file {dish_well} è minore di {conf.num_minimum_frames}: {len(frame_files)}")

    # Intorno al frame 290 ho sempre picco dovuto al cambio terreno (giorno 3 se frame ogni 15 min)
    # Invece a 261/262 ho un mini shift di immagini (senza apparente motivo), quindi taglio ancora prima
    # Non prendo i primi n frame (5) perché spesso possono presentare dei problemi, come sfocature o traslazioni 
    # ingiustificate e potrebbero essere un motivo di confondimento per i modelli
    frame_files = frame_files[conf.num_initial_frames_to_cut:utils.num_frames_3Days]

    # Read the first frame
    prev_frame = cv2.imread(os.path.join(folder_path, frame_files[0]), cv2.IMREAD_GRAYSCALE)
    if prev_frame.shape[:2] != target_size:
        raise InvalidImageSizeError(f"L'immagine {frame_files[0]} non è di dimensione 500x500")
    prev_frame = cv2.resize(prev_frame, target_size)

    # METRICHE
    # media della magnitudine in ogni frame (1 valore per frame)
    # media della magnitudine tra un numero di frame specifico in avanti nel tempo (3)
    # vorticity media in ogni frame (1 valore per ogni frame che indica la media delle fasi delle coordinate polari (rotazione media))
    # media dell'inverso della deviazione standard delle direzioni dei vettori
    # hybrid: insieme di mean magnitude e st.dev (quello inverso medio calcolato prima. Non specifica come)
    mean_magnitude = []
    vorticity = []
    std_dev = []
    hybrid = []

    for frame_file in frame_files[1:]:

        # Read the current frame
        current_frame = cv2.imread(os.path.join(folder_path, frame_file), cv2.IMREAD_GRAYSCALE)
        if current_frame.shape[:2] != target_size:
            raise InvalidImageSizeError(f"L'immagine {frame_file} non è di dimensione 500x500")
        current_frame = cv2.resize(current_frame, target_size)

        # Compute optical flow using Farneback method
        magnitude, angle_degrees, flow = compute_optical_flowFarneback(prev_frame, current_frame)

        # Calculate metrics
        mean_magnitude.append(np.mean(magnitude))
        vorticity.append(calculate_vorticity(flow))
        std_dev.append(np.std(angle_degrees))

        # Calculate normalized metrics
        norm_mean_mag = (np.mean(magnitude) - np.min(mean_magnitude)) / (np.max(mean_magnitude) - np.min(mean_magnitude))
        norm_inv_std_dev = (1 / np.std(angle_degrees) - np.min(1 / np.array(std_dev))) / (np.max(1 / np.array(std_dev)) - np.min(1 / np.array(std_dev)))

        # Compute hybrid metric
        hybrid_metric = (norm_mean_mag + norm_inv_std_dev) / 2
        hybrid.append(hybrid_metric)

        # Update the previous frame for the next iteration
        prev_frame = current_frame

    sum_mean_mag = []

    for frame in range(0, len(mean_magnitude)-conf.num_forward_frame+1):
        sum_mean_mag.append(np.mean(mean_magnitude[frame:frame+conf.num_forward_frame]))
    
    padding_length = len(mean_magnitude) - len(sum_mean_mag)
    sum_mean_mag = np.pad(sum_mean_mag, ((0, padding_length)), 'constant', constant_values=(0))

    mean_magnitude = np.array(mean_magnitude).astype(float)
    vorticity = np.array(vorticity).astype(float)
    std_dev = np.array(std_dev).astype(float)
    hybrid = np.array(hybrid).astype(float)
    sum_mean_mag = np.array(sum_mean_mag).astype(float)
    print(f"Processed frames {dish_well} saved successfully")

    return mean_magnitude, vorticity, hybrid, sum_mean_mag

def main():
    # Success/Error
    n_video = 0
    n_video_blasto = 0
    n_video_no_blasto = 0
    n_video_error_blasto = 0
    n_video_error_no_blasto = 0
    n_video_success_blasto = 0
    n_video_success_no_blasto = 0

    # Initialize dicts to store metrics for all videos
    mean_magnitude_dict = {}
    vorticity_dict = {}
    hybrid_dict = {}
    sum_mean_mag_dict = {}

    for class_sample in ['blasto','no_blasto']:
        path_all_folders = myPaths.path_BlastoData + class_sample
        total_videos = len(os.listdir(path_all_folders))
        errors = []

        for sample in os.listdir(path_all_folders):
            n_video += 1
            if class_sample == "blasto":
                n_video_blasto += 1
            else:
                n_video_no_blasto += 1

            try:
                sample_path = os.path.join(path_all_folders, sample)

                mean_magnitude, vorticity, hybrid, sum_mean_mag = process_frames(sample_path, sample)
                
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
                else:
                    n_video_error_no_blasto += 1
                print('-------------------')
                print(f"Error in sample: {sample}")
                print(e)
                errors.append(sample)
                print('-------------------')
                continue
                
        print(f'Errors in {class_sample}:', errors)

    print('===================')
    print('===================')
    print(f"Sono stati analizzati {n_video} Time Lapse di cui {n_video_blasto} blasto e {n_video_no_blasto} no_blasto")
    print(f"Sono state elaborate con successo {n_video_success_blasto} serie temporali per blasto e {n_video_success_no_blasto} per no_blasto")
    print(f"Non sono state elaborate le serie temporali di {n_video_error_blasto} video blasto e {n_video_error_no_blasto} video no_blasto")

    temporal_data_directory = os.path.join(parent_dir, '_02_temporalData')

    with open(os.path.join(temporal_data_directory, 'mean_magnitude_dict.pkl'), 'wb') as mm:
        pickle.dump(mean_magnitude_dict, mm)

    with open(os.path.join(temporal_data_directory, 'vorticity_dict.pkl'), 'wb') as v:
        pickle.dump(vorticity_dict, v)

    with open(os.path.join(temporal_data_directory, 'hybrid_dict.pkl'), 'wb') as h:
        pickle.dump(hybrid_dict, h)

    with open(os.path.join(temporal_data_directory, 'sum_mean_mag_dict.pkl'), 'wb') as smm:
        pickle.dump(sum_mean_mag_dict, smm)

if __name__ == "__main__":
    execution_time = timeit.timeit(main, number=1)
    print("Tempo impiegato:", execution_time, "secondi")
