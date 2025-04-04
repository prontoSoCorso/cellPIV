import os
import cv2
import numpy as np
import logging
#logging.basicConfig(filename=os.path.join(os.path.dirname(os.path.abspath(__file__)) , 'optical_flow_errors.log'), level=logging.ERROR)

from opticalFlow_functions import (calculate_vorticity, 
                                   sort_files_by_slice_number, 
                                   compute_optical_flowPyrLK, 
                                   compute_optical_flowFarneback,
                                   preprocess_frame,
                                   save_plot_temporal_metrics,
                                   overlay_arrows)

class InsufficientFramesError(Exception):
    """Eccezione sollevata quando il numero di frame è insufficiente."""
    pass

class InvalidImageSizeError(Exception):
    """Eccezione sollevata quando un'immagine non ha la dimensione 500x500."""
    pass

class InvalidOpticalFlowMethodError(Exception):
    """Eccezione sollevata quando il metodo scelto di flusso ottico non è LK o Farneback."""

def process_frames(folder_path, 
                   dish_well, 
                   img_size, 
                   num_minimum_frames, 
                   num_initial_frames_to_cut, 
                   num_forward_frame, 
                   method_optical_flow,
                   save_metrics=False,
                   output_metrics_base_path="", 
                   save_overlay_optical_flow=False,
                   output_path_images_with_optical_flow=""
                   ):
    """Main processing function with enhanced GPU support and error handling"""
    target_size = (img_size, img_size)  # Imposto la dimensione delle immagini

    # Ottengo l'elenco dei frame ordinati
    frame_files = sort_files_by_slice_number(os.listdir(folder_path))

    # Controllo se il numero di frame è sufficiente
    if len(frame_files) < num_minimum_frames:
        raise InsufficientFramesError(f"Frame number too low for: {dish_well}. \nIt must have more than {num_minimum_frames} frames. It has {len(frame_files)} frames")

    # Rimuovo i primi "num_initial_frames_to_cut" (e.g. causa spostamenti/motivi biologici)
    frame_files = frame_files[num_initial_frames_to_cut:]

    # Leggo il primo frame
    prev_frame = cv2.imread(os.path.join(folder_path, frame_files[0]), cv2.IMREAD_GRAYSCALE)
    if prev_frame.shape[:2] != target_size:
        prev_frame = cv2.resize(prev_frame, target_size)
    prev_frame = preprocess_frame(prev_frame)

    # METRICHE
    # media della magnitudine in ogni frame (1 valore per frame)
    # media della magnitudine tra un numero di frame specifico in avanti nel tempo (3)
    # vorticity media in ogni frame (1 valore per ogni frame che indica la media delle fasi delle coordinate polari (rotazione media))
    # media dell'inverso della deviazione standard delle direzioni dei vettori
    # hybrid: insieme di mean magnitude e st.dev (quello inverso medio calcolato prima. Non specifica come)
    metrics = {
        'mean_magnitude': [],
        'vorticity': [],
        'std_dev': [],
        'hybrid': [],
        'sum_mean_mag': []
    }

    for frame_idx, frame_file in enumerate(frame_files[1:], 1):
        try:
            # Leggo il frame corrente
            current_frame_path = os.path.join(folder_path, frame_file)
            current_frame = cv2.imread(current_frame_path, cv2.IMREAD_GRAYSCALE)
            if current_frame.shape[:2] != target_size:
                current_frame = cv2.resize(current_frame, target_size)
            current_frame = preprocess_frame(current_frame)

            # Scelgo il metodo di Optical Flow
            if method_optical_flow == "LucasKanade":
                magnitude, angle_degrees, flow, _ = compute_optical_flowPyrLK(prev_frame, current_frame)
            elif method_optical_flow == "Farneback":
                magnitude, angle_degrees, flow = compute_optical_flowFarneback(prev_frame, current_frame)
            else:
                raise InvalidOpticalFlowMethodError(f"Il metodo selezionato, {method_optical_flow}, non è implementato")

            # Calcolo le metriche
            metrics['mean_magnitude'].append(np.mean(magnitude))
            metrics['vorticity'].append(calculate_vorticity(flow))
            # Convert angles to complex unit vectors
            angles_rad = np.deg2rad(angle_degrees)
            mean_vector = np.exp(1j * angles_rad).mean()
            angular_deviation = 1 - np.abs(mean_vector)  # 0=aligned, 1=random directions
            
            metrics['std_dev'].append(angular_deviation)

            # Visualization
            if save_overlay_optical_flow:
                frame_viz = current_frame.copy()
                frame_viz = overlay_arrows(frame_viz, flow)  # Add arrows to copy
                cv2.imwrite(os.path.join(output_path_images_with_optical_flow, f"frame_{frame_idx:03d}.png"), frame_viz)
        
            # Aggiorno il frame precedente
            prev_frame = current_frame

        except Exception as e:
            logging.error(f"Error in {dish_well} at frame {frame_idx} ({frame_file}): {str(e)}")

    # Global normalization for metrics
    mean_mag = np.array(metrics['mean_magnitude'])
    norm_mag = (mean_mag - mean_mag.min()) / (mean_mag.max() - mean_mag.min() + 1e-8)
    std_dev = np.array(metrics['std_dev'])
    norm_std = (std_dev - std_dev.min()) / (std_dev.max() - std_dev.min() + 1e-8)
    
    metrics['hybrid'] = ((norm_mag + (1 - norm_std)) / 2).tolist()  # Inverted std relationship
    
    # Calcolo sum_mean_mag
    sum_mean_mag = [np.mean(mean_mag[i:i + num_forward_frame]) 
                    for i in range(len(mean_mag) - num_forward_frame + 1)]
    
    # Padding per mantenere la lunghezza uguale
    sum_mean_mag = [np.mean(mean_mag[i:i+num_forward_frame]) 
                   for i in range(len(mean_mag) - num_forward_frame + 1)]
    padding_length = len(mean_mag) - len(sum_mean_mag)
    metrics['sum_mean_mag'] = np.pad(sum_mean_mag, 
                                     ((0, padding_length)), 
                                     'constant', 
                                     constant_values=(0))

    logging.info(f"Processed frames {dish_well} saved successfully")

    if save_metrics:
        save_plot_temporal_metrics(output_base=output_metrics_base_path, 
                                   dish_well=dish_well, 
                                   metrics=metrics,
                                   start_frame=num_initial_frames_to_cut)

    return metrics
