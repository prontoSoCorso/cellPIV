import os
import pickle
import cv2
import numpy as np
import timeit
from myFunctions import calculate_vorticity, sort_files_by_slice_number, compute_optical_flowPyrLK, compute_optical_flowFarneback

import sys
# Configurazione dei percorsi e dei parametri
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_01_OpticalFlow as conf
from config import user_paths as myPaths
from config import utils as utils

class InsufficientFramesError(Exception):
    """Eccezione sollevata quando il numero di frame è insufficiente."""
    pass

class InvalidImageSizeError(Exception):
    """Eccezione sollevata quando un'immagine non ha la dimensione 500x500."""
    pass

class InvalidOpticalFlowMethodError(Exception):
    """Eccezione sollevata quando il metodo scelto di flusso ottico non è LK o Farneback"""
    SystemExit()

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
    frame_files = frame_files[conf.num_initial_frames_to_cut:]

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

        if conf.method_optical_flow == "LucasKanade":
            # Compute optical flow
            magnitude, angle_degrees, flow, _   = compute_optical_flowPyrLK(prev_frame, current_frame)
        elif conf.method_optical_flow == "Farneback":
            magnitude, angle_degrees, flow      = compute_optical_flowFarneback(prev_frame, current_frame)
        else:
            raise InvalidOpticalFlowMethodError(f" Il metodo selezionato, {conf.method_optical_flow} non è implementato")

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

    # Compute sum mean mag
    sum_mean_mag = []

    for frame in range(0, len(mean_magnitude)-conf.num_forward_frame+1):
        sum_mean_mag.append(np.mean(mean_magnitude[frame:frame+conf.num_forward_frame]))
    
    # Pad sum_mean_mag with zeros to match mean_magnitude length
    padding_length = len(mean_magnitude) - len(sum_mean_mag)
    sum_mean_mag = np.pad(sum_mean_mag, ((0, padding_length)), 'constant', constant_values=(0))

    mean_magnitude = np.array(mean_magnitude).astype(float)     # Devo convertirlo perché è una lista in partenza
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
    errors_blasto = []
    errors_no_blasto = []

    # Initialize dicts to store metrics for all videos
    mean_magnitude_dict = {}
    vorticity_dict = {}
    hybrid_dict = {}
    sum_mean_mag_dict = {}
    print(f"\n===== Metodo utilizzato per il calcolo del flusso ottico: {conf.method_optical_flow} =====\n")

    for class_sample in ['blasto','no_blasto']:
        path_all_folders = myPaths.path_BlastoData + class_sample
        total_videos = len(os.listdir(path_all_folders))

        for idx, sample in enumerate(os.listdir(path_all_folders), start=1):
            n_video += 1
            if class_sample == "blasto":
                n_video_blasto += 1
            else:
                n_video_no_blasto += 1

            print(f"Calcolando il flusso ottico del video {idx}/{total_videos} di {'blasto' if class_sample == 'blasto' else 'no_blasto'}")

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
                    errors_blasto.append(sample)
                else:
                    n_video_error_no_blasto += 1
                    errors_no_blasto.append(sample)
                print('-------------------')
                print(f"Error in sample: {sample}")
                print(e)
                print('-------------------')
                continue
        
        print('===================')
        print("Terminata Elaborazione...")
        print('===================')

    # Stampo quanti frame con successo e quanti errori
    print('===================')
    print('===================')
    print(f"Sono stati analizzati {n_video} Time Lapse di cui {n_video_blasto} blasto e {n_video_no_blasto} no_blasto")
    print(f"Sono state elaborate con successo {n_video_success_blasto} serie temporali per blasto e {n_video_success_no_blasto} per no_blasto")
    print(f"Non sono state elaborate le serie temporali di {n_video_error_blasto} video blasto e {n_video_error_no_blasto} video no_blasto")

    print('===================')
    print('===================')
    print(f'Errors in "blasto":', errors_blasto)

    print('===================')
    print('===================')
    print(f'Errors in "no_blasto":', errors_no_blasto)
    
    # Ottengo il percorso della cartella "_02_temporalData"
    temporal_data_directory = os.path.join(parent_dir, '_02_temporalData')

    # Salvataggio della lista come file utilizzando pickle nella cartella corrente
    for dict_name, dict_data in zip(
        ['mean_magnitude_dict', 'vorticity_dict', 'hybrid_dict', 'sum_mean_mag_dict'],
        [mean_magnitude_dict, vorticity_dict, hybrid_dict, sum_mean_mag_dict]):
        file_path = os.path.join(temporal_data_directory, f"{dict_name}_{conf.method_optical_flow}.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(dict_data, f)

if __name__ == "__main__":
    # Misura il tempo di esecuzione della funzione main()
    execution_time = timeit.timeit(main, number=1)
    print("Tempo impiegato:", execution_time, "secondi")