import random
import os
import cv2
import numpy as np
import sys
import logging

# Specify the desired log file path
logging.basicConfig(filename=os.path.join(os.path.dirname(os.path.abspath(__file__)) , 'optical_flow_viz_info.log'), 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Configurazione dei percorsi e dei parametri
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path) 
parent_dir = current_dir
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from _01_opticalFlows.opticalFlow_functions import (
    calculate_vorticity,
    preprocess_frame,
    sort_files_by_slice_number, 
    compute_optical_flowPyrLK, 
    compute_optical_flowFarneback)

from config import Config_01_OpticalFlow as conf
from config import user_paths as myPaths
from config import utils as utils

class InsufficientFramesError(Exception):
    pass

class InvalidImageSizeError(Exception):
    pass


def process_frames(folder_path, 
                   dish_well, 
                   img_size, 
                   num_minimum_frames, 
                   num_initial_frames_to_cut, 
                   num_forward_frame, 
                   method_optical_flow, 
                   output_metrics_base_path, 
                   output_path_images_with_optical_flow):
    """Main processing function with enhanced GPU support and error handling"""
    target_size = (img_size, img_size)

    frame_files = sort_files_by_slice_number(os.listdir(folder_path))
    
    if len(frame_files) < num_minimum_frames:
        raise InsufficientFramesError(f"Frame insufficienti per {dish_well}: {len(frame_files)}")
    
    frame_files = frame_files[num_initial_frames_to_cut:]
    prev_frame = cv2.imread(os.path.join(folder_path, frame_files[0]), cv2.IMREAD_GRAYSCALE)
    prev_frame = cv2.resize(prev_frame, target_size)
    prev_frame = preprocess_frame(prev_frame, method_optical_flow)
    
    metrics = {
        'mean_magnitude': [],
        'vorticity': [],
        'std_dev': [],
        'hybrid': [],
        'sum_mean_mag': []
    }
    all_angles = []
    
    for frame_idx, frame_file in enumerate(frame_files[1:]):
        current_frame_path = os.path.join(folder_path, frame_file)
        current_frame = cv2.imread(current_frame_path, cv2.IMREAD_GRAYSCALE)
        current_frame = cv2.resize(current_frame, target_size)
        current_frame = preprocess_frame(current_frame, method_optical_flow)
        
        # GPU-accelerated optical flow
        if method_optical_flow == "LucasKanade":
            magnitude, angle_degrees, flow, _ = compute_optical_flowPyrLK(prev_frame, current_frame)
        elif method_optical_flow == "Farneback":
            magnitude, angle_degrees, flow = compute_optical_flowFarneback(prev_frame, current_frame)
        else:
            raise InvalidOpticalFlowMethodError(f"Invalid Method: {method_optical_flow}")
        
        # Metric calculations
        metrics['mean_magnitude'].append(np.nanmean(magnitude))
        metrics['vorticity'].append(calculate_vorticity(flow))
        # Convert angles to complex unit vectors
        angles_rad = np.deg2rad(angle_degrees)
        all_angles.append(angles_rad)  # Store for later processing

        # Visualization
        if method_optical_flow == "Farneback":
            frame_viz = (current_frame.copy() * 255).clip(0, 255).astype(np.uint8)
        else:
            frame_viz = current_frame.copy()

        frame_viz = overlay_arrows(frame_viz, flow)  # Add arrows to copy
        cv2.imwrite(os.path.join(output_path_images_with_optical_flow, f"frame_{frame_idx:03d}.png"), frame_viz)
        
        prev_frame = current_frame
    
    # Post-process circular statistics
    angles_stack = np.stack(all_angles)
    mean_vector = np.exp(1j * angles_stack).mean(axis=(1,2)) # Scalar per frame
    angular_deviation = 1 - np.abs(mean_vector)
    metrics['std_dev'] = angular_deviation.tolist() # 1D (N_frames,)

    # Global normalization for metrics
    mean_mag = np.array(metrics['mean_magnitude'])
    norm_mag = (mean_mag - mean_mag.min()) / (mean_mag.max() - mean_mag.min() + 1e-8)
    
    std_dev = np.array(metrics['std_dev'])
    norm_std = (std_dev - std_dev.min()) / (std_dev.max() - std_dev.min() + 1e-8)
    
    metrics['hybrid'] = ((norm_mag + (1 - norm_std)) / 2).tolist()  # Inverted std relationship
    
    # Sum mean mag
    sum_mean_mag = [np.mean(mean_mag[i:i+num_forward_frame]) 
                   for i in range(len(mean_mag) - num_forward_frame + 1)]
    padding_length = len(mean_mag) - len(sum_mean_mag)
    metrics['sum_mean_mag'] = np.pad(sum_mean_mag,
                                    (0, padding_length),
                                    'constant')

    process_and_save_metrics(output_metrics_base_path, dish_well, metrics)
        

def main():
    n_video_target = 3  # Numero di video per ciascuna classe
    selected_videos = {"blasto": set(), "no_blasto": set()}  # Set per evitare duplicati

    logging.info(f"Starting analysis with method: {conf.method_optical_flow}")
    
    for class_sample in ['blasto', 'no_blasto']:
        path_all_folders = os.path.join(myPaths.path_BlastoData, class_sample)
        all_videos = os.listdir(path_all_folders)

        while len(selected_videos[class_sample]) < n_video_target:
            sample = random.choice(all_videos)  # Selezione casuale di un video
            if sample in selected_videos[class_sample]:  # Evita duplicati
                continue

            logging.info(f"Processing {class_sample}/{sample}")

            sample_path = os.path.join(path_all_folders, sample)
            output_base = os.path.join(current_dir, "metrics_examples", class_sample) # blablabla/_01_opticalFlows/metrics_examples/blasto oppure no_blasto
            os.makedirs(output_base, exist_ok=True)
            output_path_images = os.path.join(conf.output_path_optical_flow_images, conf.method_optical_flow, class_sample, sample)
            os.makedirs(output_path_images, exist_ok=True)

            process_frames(
                folder_path=sample_path, 
                dish_well=sample, 
                img_size=utils.img_size, 
                num_minimum_frames=conf.num_minimum_frames, 
                num_initial_frames_to_cut=conf.num_initial_frames_to_cut, 
                num_forward_frame=conf.num_forward_frame, 
                method_optical_flow=conf.method_optical_flow, 
                output_metrics_base_path=output_base, 
                output_path_images_with_optical_flow=output_path_images
            )
            
            selected_videos[class_sample].add(sample)  # Aggiungi video processato con successo
        
    print(f"\n✅ Analisi completata per {n_video_target} video 'blasto' e {n_video_target} video 'no_blasto'.")
    logging.info("Analysis completed successfully")
    
if __name__ == "__main__":
    main()
