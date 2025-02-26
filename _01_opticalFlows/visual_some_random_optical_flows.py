import random
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

from config import Config_01_OpticalFlow as conf
from config import user_paths as myPaths
from config import utils as utils

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from _01_opticalFlows.opticalFlow_functions import (
    calculate_vorticity, sort_files_by_slice_number, 
    compute_optical_flowPyrLK, compute_optical_flowFarneback
)

class InsufficientFramesError(Exception):
    pass

class InvalidImageSizeError(Exception):
    pass

class InvalidOpticalFlowMethodError(Exception):
    pass

def overlay_arrows(image, flow, step=10):
    """Sovrappone le frecce del flusso ottico sui frame."""
    h, w = image.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1)
    fx, fy = flow[y, x].T
    
    for (xi, yi, fxi, fyi) in zip(x, y, fx, fy):
        cv2.arrowedLine(image, (xi, yi), (int(xi + fxi), int(yi + fyi)), (0, 255, 0), 1, tipLength=0.3)
    return image
 

def process_and_save_metrics(output_base, dish_well, mean_magnitude, vorticity, std_dev, hybrid, sum_mean_mag):
    """Genera e salva un grafico con le metriche calcolate."""
        
    peaks_mag, _ = find_peaks(mean_magnitude, distance=10)
    peaks_vort, _ = find_peaks(np.abs(vorticity), distance=10)
    peaks_std_dev, _ = find_peaks(std_dev, distance=10)
    peaks_hybrid, _ = find_peaks(hybrid, distance=10)
    peaks_sum_mean_mag, _ = find_peaks(sum_mean_mag, distance=10)
    
    # Trovo i valori e gli indici dei 5 picchi più alti
    mean_magnitude = np.array(mean_magnitude).astype(float)     # Devo convertirlo perché è una lista in partenza
    vorticity = np.array(vorticity).astype(float)
    std_dev = np.array(std_dev).astype(float)    
    hybrid = np.array(hybrid).astype(float)
    sum_mean_mag = np.array(sum_mean_mag).astype(float)

    top_5_peaks_indices_mag = peaks_mag[np.argsort(mean_magnitude[peaks_mag])[-5:]]     # Indici dei 5 picchi più alti
    top_5_peaks_values_mag = mean_magnitude[top_5_peaks_indices_mag]                    # Valori dei 5 picchi più alti

    top_5_peaks_indices_vort = peaks_vort[np.argsort(abs(vorticity[peaks_vort]))[-5:]]    
    top_5_peaks_values_vort = vorticity[top_5_peaks_indices_vort]                    

    top_5_peaks_indices_std_dev = peaks_std_dev[np.argsort(std_dev[peaks_std_dev])[-5:]]  
    top_5_peaks_values_std_dev = std_dev[top_5_peaks_indices_std_dev]

    top_5_peaks_indices_hybrid = peaks_hybrid[np.argsort(hybrid[peaks_hybrid])[-5:]]  
    top_5_peaks_values_hybrid = hybrid[top_5_peaks_indices_hybrid]

    top_5_peaks_indices_sum_mean_mag = peaks_sum_mean_mag[np.argsort(sum_mean_mag[peaks_sum_mean_mag])[-5:]]  
    top_5_peaks_values_sum_mean_mag = sum_mean_mag[top_5_peaks_indices_sum_mean_mag]

    # Temporal Analysis (Plotting metrics over time)
    time_steps = np.arange(len(mean_magnitude))
    fig, axes = plt.subplots(5, 1, figsize=(12, 10))
    data_labels = [
        (mean_magnitude, 'Mean Magnitude', 'Magnitude', top_5_peaks_indices_mag, top_5_peaks_values_mag, 'blue', 'Magnitude'),
        (vorticity, 'Vorticity', 'Vorticity', top_5_peaks_indices_vort, top_5_peaks_values_vort, 'red', 'Vorticity'),
        (std_dev, 'Standard Deviation', 'Standard Deviation', top_5_peaks_indices_std_dev, top_5_peaks_values_std_dev, 'purple', 'Standard Deviation'),
        (hybrid, 'Hybrid', 'Hybrid', top_5_peaks_indices_hybrid, top_5_peaks_values_hybrid, 'orange', 'Hybrid'),
        (sum_mean_mag, 'Sum Mean Magnitude', 'Sum Mean Magnitude', top_5_peaks_indices_sum_mean_mag, top_5_peaks_values_sum_mean_mag, 'green', 'Sum Mean Magnitude')
    ]
    
    for i, (data, label, ylabel, peak_indices, peak_values, peak_color, peak_label) in enumerate(data_labels):
        axes[i].plot(time_steps, data, color=peak_color, label=label)
        axes[i].scatter(peak_indices, peak_values, color=peak_color, marker='o', facecolors='none', label=f'Peaks')
        for j, index_peak in enumerate(peak_indices):
            axes[i].text(index_peak, peak_values[j], str(index_peak), fontsize=9, va='bottom', ha='right' if i < 2 else 'left', color=peak_color)
        axes[i].set_xlabel('Frame Index')
        axes[i].set_ylabel(ylabel)
        axes[i].legend()
        axes[i].set_title(f"Temporal Analysis of {label}")
    
    plt.tight_layout()

    plt.savefig(os.path.join(output_base, f"{dish_well}.png"))
    plt.close()



def process_frames(folder_path, dish_well, img_size, num_minimum_frames, num_initial_frames_to_cut, num_forward_frame, method_optical_flow, output_base, output_path_optical_flow_images):
    target_size = (img_size, img_size)
    frame_files = sort_files_by_slice_number(os.listdir(folder_path))
    
    if len(frame_files) < num_minimum_frames:
        raise InsufficientFramesError(f"Frame insufficienti per {dish_well}: {len(frame_files)}")
    
    frame_files = frame_files[num_initial_frames_to_cut:]
    prev_frame = cv2.imread(os.path.join(folder_path, frame_files[0]), cv2.IMREAD_GRAYSCALE)
    if prev_frame.shape[:2] != target_size:
        raise InvalidImageSizeError(f"Dimensione errata per {frame_files[0]}")
    prev_frame = cv2.resize(prev_frame, target_size)
    
    mean_magnitude, vorticity, std_dev, hybrid = [], [], [], []
    
    for i, frame_file in enumerate(frame_files[1:]):
        current_frame_path = os.path.join(folder_path, frame_file)
        current_frame = cv2.imread(current_frame_path, cv2.IMREAD_GRAYSCALE)
        if current_frame.shape[:2] != target_size:
            raise InvalidImageSizeError(f"Dimensione errata per {frame_file}")
        current_frame = cv2.resize(current_frame, target_size)
        
        if method_optical_flow == "LucasKanade":
            magnitude, angle_degrees, flow, _ = compute_optical_flowPyrLK(prev_frame, current_frame)
        elif method_optical_flow == "Farneback":
            magnitude, angle_degrees, flow = compute_optical_flowFarneback(prev_frame, current_frame)
        else:
            raise InvalidOpticalFlowMethodError(f"Metodo {method_optical_flow} non implementato")
        
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

        # Compute hybrid metric
        hybrid_metric = (norm_mean_mag + norm_inv_std_dev) / 2
        hybrid.append(hybrid_metric)
        
        frame_with_arrows = overlay_arrows(current_frame.copy(), flow)
        cv2.imwrite(os.path.join(output_path_optical_flow_images, f"frame_{i+1}.png"), frame_with_arrows)
        
        prev_frame = current_frame
    
    sum_mean_mag = [np.mean(mean_magnitude[i:i + num_forward_frame]) for i in range(len(mean_magnitude) - num_forward_frame + 1)]
    sum_mean_mag = np.pad(sum_mean_mag, ((0, len(mean_magnitude) - len(sum_mean_mag))), 'constant', constant_values=(0))
    
    process_and_save_metrics(output_base, dish_well, mean_magnitude, vorticity, std_dev, hybrid, sum_mean_mag)





def main():
    n_video_target = 3  # Numero di video per ciascuna classe
    selected_videos = {"blasto": set(), "no_blasto": set()}  # Set per evitare duplicati

    print(f"\n===== Metodo utilizzato per il calcolo del flusso ottico: {conf.method_optical_flow} =====\n")

    for class_sample in ['blasto', 'no_blasto']:
        path_all_folders = os.path.join(myPaths.path_BlastoData, class_sample)
        all_videos = os.listdir(path_all_folders)

        while len(selected_videos[class_sample]) < n_video_target:
            sample = random.choice(all_videos)  # Selezione casuale di un video
            if sample in selected_videos[class_sample]:  # Evita duplicati
                continue

            print(f"Calcolando il flusso ottico per {class_sample}: {sample}")
            sample_path = os.path.join(path_all_folders, sample)
            output_base = os.path.join(current_dir, "metrics_examples", class_sample) # blablabla/_01_opticalFlows/metrics_examples/blasto oppure no_blasto
            os.makedirs(output_base, exist_ok=True)
            output_path_images = os.path.join(conf.output_path_optical_flow_images, class_sample, sample)
            os.makedirs(output_path_images, exist_ok=True)

            try:
                process_frames(
                    folder_path=sample_path, dish_well=sample, img_size=utils.img_size, 
                    num_minimum_frames=conf.num_minimum_frames, num_initial_frames_to_cut=conf.num_initial_frames_to_cut, 
                    num_forward_frame=conf.num_forward_frame, method_optical_flow=conf.method_optical_flow, 
                    output_base=output_base, output_path_optical_flow_images=output_path_images
                )
                
                selected_videos[class_sample].add(sample)  # Aggiungi video processato con successo
                print(f"✅ Video {sample} processato correttamente.")
            
            except Exception as e:
                print(f"❌ Errore nel sample {sample}: {e}")
                print("Selezionando un altro video...\n")
                continue

    print(f"\n✅ Analisi completata per {n_video_target} video 'blasto' e {n_video_target} video 'no_blasto'.")
    
    
if __name__ == "__main__":
    main()
