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
from config import utils as utils
from _01_opticalFlows._process_optical_flow import process_frames
from _utils_._utils import config_logging

def main(input_path,
         output_base_dir,
         method_optical_flow=config.method_optical_flow,
         img_size=config.img_size,
         num_minimum_frames=config.num_minimum_frames,
         num_initial_frames_to_cut=config.num_initial_frames_to_cut,
         num_forward_frame=config.num_forward_frame,
         save_metrics=config.save_metrics,
         save_overlay_optical_flow=config.save_overlay_optical_flow,
         save_final_data=config.save_final_data):
    """
    Process a single folder (input_path) and save:
      - overlay images with optical flow in output_base_dir/overlays
      - temporal metrics plots in output_base_dir/metrics
      - pickled metrics arrays (optional)
    """
    # Configure logging
    os.makedirs(output_base_dir, exist_ok=True)
    log_file = os.path.join(output_base_dir, f'optical_flow_{method_optical_flow}.log')
    config_logging(log_dir=output_base_dir, log_filename=os.path.basename(log_file))

    logging.info(f"Starting optical flow analysis on: {input_path}")
    logging.info(f"Method: {method_optical_flow}, img_size: {img_size}, min_frames: {num_minimum_frames}, "
                 f"cut_initial: {num_initial_frames_to_cut}, forward_frames: {num_forward_frame}")

    # Prepara cartelle di output
    overlays_dir = os.path.join(output_base_dir, "overlays", method_optical_flow)
    metrics_plots_dir = os.path.join(output_base_dir, "metrics_plots", method_optical_flow)
    temporal_data_dir = os.path.join(output_base_dir, "temporal_data", method_optical_flow)
    os.makedirs(overlays_dir, exist_ok=True)
    os.makedirs(metrics_plots_dir, exist_ok=True)
    os.makedirs(temporal_data_dir, exist_ok=True)

    try:
        # Processa i frames
        metrics = process_frames(
            folder_path=input_path,
            dish_well=os.path.basename(input_path),
            img_size=img_size,
            num_minimum_frames=num_minimum_frames,
            num_initial_frames_to_cut=num_initial_frames_to_cut,
            num_forward_frame=num_forward_frame,
            method_optical_flow=method_optical_flow,
            save_metrics=save_metrics,
            output_metrics_base_path=metrics_plots_dir,
            save_overlay_optical_flow=save_overlay_optical_flow,
            output_path_images_with_optical_flow=overlays_dir,

            # parametri Farneback / LK
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

        # Salva i dati grezzi delle metriche
        if save_final_data:
            pickle_path = os.path.join(temporal_data_dir, f"{os.path.basename(input_path)}_{method_optical_flow}.pkl")
            with open(pickle_path, 'wb') as f:
                pickle.dump(metrics, f)
            logging.info(f"Pickle metrics saved to {pickle_path}")

        logging.info("Processing completed successfully")

    except Exception as e:
        logging.error(f"Error processing folder {input_path}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Optical Flow analysis per singola cartella di frames"
    )
    parser.add_argument("input_path",
                        help="Path alla cartella contenente i frames da processare")
    parser.add_argument("output_base_dir",
                        help="Directory base per salvare overlay, grafici e dati")
    parser.add_argument("--method", default=conf.method_optical_flow,
                        choices=["Farneback", "LucasKanade"],
                        help="Metodo di Optical Flow")
    args = parser.parse_args()
    """

    start = time.time()
    
    video_name = "D2020.08.25_S02605_I0141_D_2"
    input_path = "/home/phd2/Scrivania/CorsoData/blastocisti/blasto/" + video_name
    output_base_dir = "/home/phd2/Scrivania/CorsoData/OpticalFlowSingleVideo/" + video_name
    method = "Farneback"
    main(input_path=input_path,
         output_base_dir=output_base_dir,
         method_optical_flow=method,
         save_metrics=True,
         save_overlay_optical_flow=True,
         save_final_data=True)
    print(f"Execution time: {time.time() - start:.2f} s")
