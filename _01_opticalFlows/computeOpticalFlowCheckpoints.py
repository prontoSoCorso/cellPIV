import os
import pickle
import cv2
import numpy as np
import timeit
from datetime import datetime, timedelta
import time
from myFunctions import calculate_vorticity, sort_files_by_slice_number, compute_optical_flowPyrLK, compute_optical_flowFarneback
from PIL import Image, ImageFile

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

# Funzione per correggere immagini troncate
def fix_truncated_jpeg(file_path):
    """
    Tenta di correggere un file JPEG troncato.
    """
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        with Image.open(file_path) as img:
            # Salva il file correggendo eventuali problemi di fine file
            corrected_path = file_path + "_fixed.jpg"
            img.save(corrected_path, "JPEG")
        return corrected_path
    except Exception as e:
        print(f"Errore nella correzione del file {file_path}: {e}")
        return None

# Funzione per salvare il checkpoint
def save_checkpoint(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

# Funzione per caricare il checkpoint
def load_checkpoint(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None

def process_frames(folder_path, dish_well):
    target_size = (utils.img_size, utils.img_size)    # Impongo dimensione immagini

    # Get the list of frames in the folder
    frame_files = sort_files_by_slice_number(os.listdir(folder_path))

    # Controllo se il numero di frame è sufficiente
    if len(frame_files) < conf.num_minimum_frames:
        raise InsufficientFramesError(f"Il numero di frame nel file {dish_well} è minore di {conf.num_minimum_frames}: {len(frame_files)}")

    # Intorno al frame 290 ho sempre picco dovuto al cambio terreno (giorno 3 se frame ogni 15 min)
    # Invece a 261/262 ho un mini shift di immagini (senza apparente motivo), quindi taglio ancora prima
    # Non prendo i primi n frame (5) perché spesso possono presentare dei problemi, come sfocature o traslazioni 
    # ingiustificate e potrebbero essere un motivo di confondimento per i modelli
    frame_files = frame_files[conf.num_initial_frames_to_cut:]

    # Read the first frame
    first_frame_path = os.path.join(folder_path, frame_files[0])
    prev_frame = cv2.imread(first_frame_path, cv2.IMREAD_GRAYSCALE)
    
    if prev_frame is None:
        corrected_path = fix_truncated_jpeg(first_frame_path)
        if corrected_path:
            prev_frame = cv2.imread(corrected_path, cv2.IMREAD_GRAYSCALE)
        if prev_frame is None:
            raise ValueError(f"Non è stato possibile leggere il frame iniziale: {first_frame_path}")

    if prev_frame.shape[:2] != target_size:
        raise InvalidImageSizeError(f"L'immagine {frame_files[0]} non è di dimensione 500x500")
    prev_frame = cv2.resize(prev_frame, target_size)

    # METRICHE
    # media della magnitudine in ogni frame (1 valore per frame)
    # media della magnitudine tra un numero di frame specifico in avanti nel tempo (3)
    # vorticity media in ogni frame (1 valore per ogni frame che indica la media delle fasi delle coordinate polari (rotazione media))
    # media dell'inverso della deviazione standard delle direzioni dei vettori
    # hybrid: insieme di mean magnitude e st.dev (quello inverso medio calcolato prima. Non specifica come)
    mean_magnitude, vorticity, std_dev, hybrid = [], [], [], []

    for frame_file in frame_files[1:]:
        # Read the current frame
        current_frame_path = os.path.join(folder_path, frame_file)
        current_frame = cv2.imread(current_frame_path, cv2.IMREAD_GRAYSCALE)
        
        if current_frame is None:
            corrected_path = fix_truncated_jpeg(current_frame_path)
            if corrected_path:
                current_frame = cv2.imread(corrected_path, cv2.IMREAD_GRAYSCALE)
            if current_frame is None:
                raise ValueError(f"Non è stato possibile leggere il frame: {current_frame_path}")
        
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
        norm_mean_mag = (np.mean(magnitude) - np.min(mean_magnitude)) / (np.max(mean_magnitude) - np.min(mean_magnitude)) if (np.max(mean_magnitude) - np.min(mean_magnitude))!=0 else 0
        norm_inv_std_dev = (1 / np.std(angle_degrees) - np.min(1 / np.array(std_dev))) / (np.max(1 / np.array(std_dev)) - np.min(1 / np.array(std_dev))) if (np.max(1 / np.array(std_dev)) - np.min(1 / np.array(std_dev)))!=0 else 0

        # Compute hybrid metric
        hybrid_metric = (norm_mean_mag + norm_inv_std_dev) / 2
        hybrid.append(hybrid_metric)

        # Update the previous frame for the next iteration
        prev_frame = current_frame

    # Compute sum mean mag
    sum_mean_mag = [np.mean(mean_magnitude[i:i + conf.num_forward_frame]) for i in range(len(mean_magnitude) - conf.num_forward_frame + 1)]
    
    # Pad sum_mean_mag with zeros to match mean_magnitude length
    padding_length = len(mean_magnitude) - len(sum_mean_mag)
    sum_mean_mag = np.pad(sum_mean_mag, ((0, padding_length)), 'constant', constant_values=(0))

    print(f"Processed frames {dish_well} saved successfully")

    return (np.array(mean_magnitude).astype(float), np.array(vorticity).astype(float),
            np.array(hybrid).astype(float), np.array(sum_mean_mag).astype(float))


def main():
    checkpoint_path = os.path.join(parent_dir, "_01_opticalFlows", "checkpoint.pkl")
    checkpoint_data = load_checkpoint(checkpoint_path)

    # Imposto l'ora per salvataggio checkpoint, n ore dall'inizio
    delta_time = 1 #hours
    end_time = datetime.now() + timedelta(hours=delta_time)
    
    if checkpoint_data:
        print('\n================================================')
        print("E' stato trovato un checkpoint da cui partire")
        print('================================================')

        # Riprendi dai dati di checkpoint
        n_video = checkpoint_data['n_video']
        n_video_blasto = checkpoint_data['n_video_blasto']
        n_video_no_blasto = checkpoint_data['n_video_no_blasto']
        n_video_error_blasto = checkpoint_data['n_video_error_blasto']
        n_video_error_no_blasto = checkpoint_data['n_video_error_no_blasto']
        n_video_success_blasto = checkpoint_data['n_video_success_blasto']
        n_video_success_no_blasto = checkpoint_data['n_video_success_no_blasto']
        mean_magnitude_dict = checkpoint_data['mean_magnitude_dict']
        vorticity_dict = checkpoint_data['vorticity_dict']
        hybrid_dict = checkpoint_data['hybrid_dict']
        sum_mean_mag_dict = checkpoint_data['sum_mean_mag_dict']
        errors = checkpoint_data['errors']
        start_class_sample = checkpoint_data['class_sample']

    else:
        print('\n===============================================================')
        print("Inizializzazione delle variabili - Nessun Checkpoint trovato")
        print('===============================================================')

        # Inizializza le variabili se non ci sono checkpoint
        n_video = 0
        n_video_blasto = 0
        n_video_no_blasto = 0
        n_video_error_blasto = 0
        n_video_error_no_blasto = 0
        n_video_success_blasto = 0
        n_video_success_no_blasto = 0
        mean_magnitude_dict = {}
        vorticity_dict = {}
        hybrid_dict = {}
        sum_mean_mag_dict = {}
        errors = []
        start_class_sample = 'blasto'

    # Stampo a video il metodo di flusso ottico utilizzato
    print(f"\n===== Metodo utilizzato per il calcolo del flusso ottico: {conf.method_optical_flow} =====\n")

    class_samples = ['blasto', 'no_blasto']
    for class_sample in class_samples[class_samples.index(start_class_sample):]:
        # Imposta il punto di partenza in base al checkpoint
        start_idx = n_video_blasto if class_sample == 'blasto' else n_video_no_blasto
    
        path_all_folders = os.path.join(myPaths.path_BlastoData, class_sample)
        total_videos = len(os.listdir(path_all_folders))

        # Riprende dal video successivo se il checkpoint è stato salvato a metà
        for idx, sample in enumerate(os.listdir(path_all_folders)[start_idx:], start=start_idx + 1):
            n_video += 1
            if class_sample == "blasto":
                n_video_blasto += 1
            else:
                n_video_no_blasto += 1

            print('-------------------')
            print(f"Calcolando il flusso ottico del video {idx}/{total_videos} di {class_sample}")

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
                print(f"Error in sample: {sample}")
                print(e)
                errors.append(sample)
                continue
            
            # Controllo il tempo trascorso e salvo eventualmente un checkpoint, se superato il limite config.expTime
            if datetime.now() >= end_time:
                checkpoint_data = {
                    'n_video': n_video,
                    'n_video_blasto': n_video_blasto,
                    'n_video_no_blasto': n_video_no_blasto,
                    'n_video_error_blasto': n_video_error_blasto,
                    'n_video_error_no_blasto': n_video_error_no_blasto,
                    'n_video_success_blasto': n_video_success_blasto,
                    'n_video_success_no_blasto': n_video_success_no_blasto,
                    'mean_magnitude_dict': mean_magnitude_dict,
                    'vorticity_dict': vorticity_dict,
                    'hybrid_dict': hybrid_dict,
                    'sum_mean_mag_dict': sum_mean_mag_dict,
                    'errors': errors,
                    'class_sample': class_sample,
                    'sample_idx': idx
                }
                save_checkpoint(checkpoint_data, checkpoint_path)
                print(f"Numero di serie temporali attualmente salvate in 'mean_magnitude_dict': {len(mean_magnitude_dict)}")
                print(f"Numero di serie temporali attualmente salvate in 'sum_mean_mag_dict': {len(sum_mean_mag_dict)}")
                print("Checkpoint salvato")
                end_time = datetime.now() + timedelta(hours=delta_time)
            
        print(f'Errors in {class_sample}:', errors)


    # Stampo quanti frame con successo e quanti errori
    print('===================')
    print('===================')
    print(f"Analizzati {n_video} Time Lapse di cui {n_video_blasto} blasto e {n_video_no_blasto} no_blasto")
    print(f"Salvati correttamente: {n_video_success_blasto} blasto e {n_video_success_no_blasto} no_blasto")
    print(f"Errori: {n_video_error_blasto} blasto e {n_video_error_no_blasto} no_blasto")

    # Ottengo il percorso della cartella "_02_temporalData"
    temporal_data_directory = os.path.join(parent_dir, '_02_temporalData', 'tmp_files')

    # Salvataggio della lista come file utilizzando pickle nella cartella corrente
    for dict_name, dict_data in zip(
        ['mean_magnitude_dict', 'vorticity_dict', 'hybrid_dict', 'sum_mean_mag_dict'],
        [mean_magnitude_dict, vorticity_dict, hybrid_dict, sum_mean_mag_dict]):
        file_path = os.path.join(temporal_data_directory, f"{dict_name}_{conf.method_optical_flow}.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(dict_data, f)

    print("\nFile dei dati temporali salvati con successo!\n")
    print(f"Numero totale di serie temporali salvate in 'mean_magnitude_dict': {len(mean_magnitude_dict)}")
    print(f"Video con errori: {errors}")

if __name__ == "__main__":
    # Misura il tempo di esecuzione della funzione main()
    execution_time = timeit.timeit(main, number=1)
    print("Tempo impiegato:", execution_time, "secondi")