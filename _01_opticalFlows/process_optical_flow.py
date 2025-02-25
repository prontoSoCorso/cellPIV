import os
import cv2
import numpy as np
import sys
# Configurazione dei percorsi e dei parametri
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path) 
parent_dir = current_dir
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from _01_opticalFlows.opticalFlow_functions import calculate_vorticity, sort_files_by_slice_number, compute_optical_flowPyrLK, compute_optical_flowFarneback

class InsufficientFramesError(Exception):
    """Eccezione sollevata quando il numero di frame è insufficiente."""
    pass

class InvalidImageSizeError(Exception):
    """Eccezione sollevata quando un'immagine non ha la dimensione 500x500."""
    pass

class InvalidOpticalFlowMethodError(Exception):
    """Eccezione sollevata quando il metodo scelto di flusso ottico non è LK o Farneback."""
    SystemExit()

def process_frames(folder_path, dish_well, img_size, num_minimum_frames, num_initial_frames_to_cut, num_forward_frame, method_optical_flow):
    target_size = (img_size, img_size)  # Imposto la dimensione delle immagini

    # Ottengo l'elenco dei frame ordinati
    frame_files = sort_files_by_slice_number(os.listdir(folder_path))

    # Controllo se il numero di frame è sufficiente
    if len(frame_files) < num_minimum_frames:
        raise InsufficientFramesError(f"Il numero di frame nel file {dish_well} è minore di {num_minimum_frames}: {len(frame_files)}")

    # Rimuovo i primi frame problematici
    frame_files = frame_files[num_initial_frames_to_cut:]

    # Leggo il primo frame
    prev_frame = cv2.imread(os.path.join(folder_path, frame_files[0]), cv2.IMREAD_GRAYSCALE)
    if prev_frame.shape[:2] != target_size:
        raise InvalidImageSizeError(f"L'immagine {frame_files[0]} non è di dimensione {img_size}x{img_size}")
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
        # Leggo il frame corrente
        current_frame_path = os.path.join(folder_path, frame_file)
        current_frame = cv2.imread(current_frame_path, cv2.IMREAD_GRAYSCALE)
        if current_frame.shape[:2] != target_size:
            raise InvalidImageSizeError(f"L'immagine {frame_file} non è di dimensione {img_size}x{img_size}")
        current_frame = cv2.resize(current_frame, target_size)

        # Scelgo il metodo di Optical Flow
        if method_optical_flow == "LucasKanade":
            magnitude, angle_degrees, flow, _ = compute_optical_flowPyrLK(prev_frame, current_frame)
        elif method_optical_flow == "Farneback":
            magnitude, angle_degrees, flow = compute_optical_flowFarneback(prev_frame, current_frame)
        else:
            raise InvalidOpticalFlowMethodError(f"Il metodo selezionato, {method_optical_flow}, non è implementato")

        # Calcolo le metriche
        mean_magnitude.append(np.mean(magnitude))
        vorticity.append(calculate_vorticity(flow))
        std_dev.append(np.std(angle_degrees))

        # Calculate normalized metrics
        denom_mean_mag = np.max(mean_magnitude) - np.min(mean_magnitude)
        if denom_mean_mag == 0:
            norm_mean_mag = 0  # valore predefinito che ha senso nel contesto
        else:
            norm_mean_mag = (np.mean(magnitude) - np.min(mean_magnitude)) / denom_mean_mag
        std_angle = np.std(angle_degrees)
        std_dev_array = 1 / np.array(std_dev)
        denom_std_dev = np.max(std_dev_array) - np.min(std_dev_array)
        if std_angle == 0 or denom_std_dev == 0:
            norm_inv_std_dev = 0
        else:
            norm_inv_std_dev = (1 / std_angle - np.min(std_dev_array)) / denom_std_dev
            
        # Calcolo la metrica ibrida
        hybrid_metric = (norm_mean_mag + norm_inv_std_dev) / 2
        hybrid.append(hybrid_metric)

        # Aggiorno il frame precedente
        prev_frame = current_frame

    # Calcolo sum_mean_mag
    sum_mean_mag = [np.mean(mean_magnitude[i:i + num_forward_frame]) for i in range(len(mean_magnitude) - num_forward_frame + 1)]
    
    # Padding per mantenere la lunghezza uguale
    padding_length = len(mean_magnitude) - len(sum_mean_mag)
    sum_mean_mag = np.pad(sum_mean_mag, ((0, padding_length)), 'constant', constant_values=(0))

    print(f"Processed frames {dish_well} saved successfully")

    return (np.array(mean_magnitude).astype(float), np.array(vorticity).astype(float),
            np.array(hybrid).astype(float), np.array(sum_mean_mag).astype(float))
