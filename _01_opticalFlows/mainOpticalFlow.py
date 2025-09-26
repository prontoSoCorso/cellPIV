import os
import pickle
import sys
import time
import logging
import numpy as np
import cv2

# Configurazione dei percorsi e dei parametri
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = current_dir
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_01_OpticalFlow as config
from config import user_paths as myPaths
from config import utils as utils
from _01_opticalFlows._process_optical_flow import process_frames
from _utils_._utils import config_logging

def main(method_optical_flow=config.method_optical_flow, path_BlastoData=myPaths.path_BlastoData, 
         img_size=config.img_size, num_minimum_frames=config.num_minimum_frames, 
         num_initial_frames_to_cut=config.num_initial_frames_to_cut, num_forward_frame=config.num_forward_frame,
         output_metrics_base_dir = os.path.join(current_dir, "metrics_examples"),
         save_metrics=config.save_metrics,
         output_path_optical_flow_images=config.output_path_optical_flow_images,
         save_overlay_optical_flow=config.save_overlay_optical_flow,
         save_final_data=config.save_final_data):
    # Configure logging
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                           f'optical_flow_complete_analysis_{method_optical_flow}.log')
    config_logging(log_dir=os.path.dirname(log_file), log_filename=os.path.basename(log_file))
    
    # Initialize counters
    counters = {
        'total': 0,
        'blasto': {'total': 0, 'success': 0, 'errors': 0, 'error_list': []},
        'no_blasto': {'total': 0, 'success': 0, 'errors': 0, 'error_list': []}
    }

    # Initialize metrics dictionaries
    metrics_dicts = {
        'mean_magnitude': {},
        'vorticity': {},
        'hybrid': {},
        'sum_mean_mag': {}
    }

    logging.info(f"Starting optical flow analysis using method: {method_optical_flow.upper()}")
    logging.info(f"Configuration parameters:\n"
                 f"- Image size: {img_size}\n"
                 f"- Minimum frames: {num_minimum_frames}\n"
                 f"- Initial frames to cut: {num_initial_frames_to_cut}\n"
                 f"- Forward frames: {num_forward_frame}")

    for class_sample in ['blasto', 'no_blasto']:
        class_path = os.path.join(path_BlastoData, class_sample)
        samples = os.listdir(class_path)
        total_videos = len(samples)
        logging.info(f"\nProcessing {class_sample.upper()} class with {total_videos} samples")

        for idx, sample in enumerate(samples, start=1):
            counters['total'] += 1
            counters[class_sample]['total'] += 1
            sample_path = os.path.join(class_path, sample)
            
            try:
                logging.info(f"[{idx}/{total_videos}] Processing {class_sample}: {sample}")
                
                # Prepare output paths
                output_metrics_path = os.path.join(output_metrics_base_dir, method_optical_flow, class_sample)
                output_images_path = os.path.join(output_path_optical_flow_images, method_optical_flow, class_sample, sample)
                os.makedirs(output_metrics_path, exist_ok=True)
                os.makedirs(output_images_path, exist_ok=True)

                # Process frames
                metrics = process_frames(
                    folder_path=sample_path, 
                    dish_well=sample, 
                    img_size=img_size, 
                    num_minimum_frames=num_minimum_frames, 
                    num_initial_frames_to_cut=num_initial_frames_to_cut, 
                    num_forward_frame=num_forward_frame, 
                    method_optical_flow=method_optical_flow,
                    output_metrics_base_path=output_metrics_path,
                    save_metrics=save_metrics,
                    save_overlay_optical_flow=save_overlay_optical_flow,
                    output_path_images_with_optical_flow=output_images_path,

                    pyr_scale           = config.pyr_scale,
                    levels              = config.levels,
                    winsize_Farneback   = config.winSize_Farneback,
                    iterations          = config.iterations,
                    poly_n              = config.poly_n,
                    poly_sigma          = config.poly_sigma,
                    flags               = config.flags,

                    winSize_LK      = config.winSize_LK,
                    maxLevelPyramid = config.maxLevelPyramid,
                    maxCorners      = config.maxCorners,
                    qualityLevel    = config.qualityLevel,
                    minDistance     = config.minDistance,
                    blockSize       = config.blockSize
                    )
                
                # Store metrics
                for metric in metrics_dicts.keys():
                    metrics_dicts[metric][sample] = np.array(metrics[metric]).astype(float)
                
                counters[class_sample]['success'] += 1
        
            except Exception as e:
                counters[class_sample]['errors'] += 1
                counters[class_sample]['error_list'].append(sample)
                logging.error(f"Error processing {class_sample} sample {sample}", exc_info=True)
                continue
        

    # Log final summary
    logging.info("\n" + "="*50)
    logging.info("Processing Summary:")
    logging.info(f"Total videos processed: {counters['total']}")
    
    for class_type in ['blasto', 'no_blasto']:
        logging.info(f"\n{class_type.upper()} Class:")
        logging.info(f"- Total samples: {counters[class_type]['total']}")
        logging.info(f"- Successful processing: {counters[class_type]['success']}")
        logging.info(f"- Failed processing: {counters[class_type]['errors']}")
        if counters[class_type]['errors'] > 0:
            logging.warning(f"- Error list: {counters[class_type]['error_list']}")

    # Save final data
    if save_final_data:
        temporal_data_dir = config.pickle_dir
        os.makedirs(temporal_data_dir, exist_ok=True)

        logging.info(f"\nSaving metrics data to: {temporal_data_dir}")
        for metric_name, metric_data in metrics_dicts.items():
            file_path = os.path.join(temporal_data_dir, f"{metric_name}_{method_optical_flow}.pkl")
            with open(file_path, 'wb') as f:
                pickle.dump(metric_data, f)
            logging.debug(f"Saved {metric_name} metrics ({len(metric_data)} entries) to {file_path}")

    logging.info("\nAnalysis completed successfully")

if __name__ == "__main__":
    start_time = time.time()
    # execution_time = timeit.timeit(main, number=1)
    main(method_optical_flow=config.method_optical_flow, path_BlastoData=myPaths.path_BlastoData, 
         img_size=config.img_size, num_minimum_frames=config.num_minimum_frames, 
         num_initial_frames_to_cut=config.num_initial_frames_to_cut, num_forward_frame=config.num_forward_frame,
         output_metrics_base_dir = os.path.join(current_dir, "metrics_examples"),
         save_metrics=config.save_metrics,
         output_path_optical_flow_images=config.output_path_optical_flow_images,
         save_overlay_optical_flow=config.save_overlay_optical_flow)
    print("Execution time:", str(time.time()-start_time), "seconds")