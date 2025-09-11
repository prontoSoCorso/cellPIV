import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import sys
# Configurazione dei percorsi e dei parametri
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import utils as utils

def sort_files_by_slice_number(file_list):
    # Two possible filename formats:

    # D2013.02.19_S0675_I141_5_7_0_1.5h.jpg
    # DYear.Month.Day_DishNumber_DishNumber_Well_Frame_equatPlane_XXh.jpg
    
    # D2013.02.19_S0675_I141_D_5_7_0_1.5h.jpg
    # DYear.Month.Day_DishNumber_DishNumber_D__Well_D0_Frame_XXh.jpg

    # Define a custom sorting key function
    def get_slice_number(filename):
        if '_D_' in filename:
            slice_number_str = filename.split('_')[5]
        else:
            slice_number_str = filename.split('_')[4]
        # Convert the extracted string to an integer
        return int(slice_number_str)
    
    # Filter files that contain the desired sequence (e.g., 'jpg') in the filename
    filtered_files = [filename for filename in file_list if 'jpg' in filename]

    # Use the sorted function with the custom key function
    sorted_files = sorted(filtered_files, key=get_slice_number)

    return sorted_files

class InvalidOpticalFlowMethodError(Exception):
    pass

"""
def overlay_arrows(frame, magnitude, angle_degrees, prev_pts):
    for i, (mag, angle) in enumerate(zip(magnitude, angle_degrees)):
        # Estraggo le coordinate x e y del flusso ottico
        x, y = prev_pts[i].ravel()
        dx, dy = mag * np.cos(np.radians(angle)), mag * np.sin(np.radians(angle))
        # Calcolo il punto finale della freccia
        endpoint = (int(x + dx[0]), int(y + dy[0]))
        # Disegno la freccia
        cv2.arrowedLine(frame, (int(x), int(y)), endpoint, (255, 0, 0), 1)
    return frame
"""

def overlay_arrows(image, flow, step=10, blur_ksize=25, blur_sigma=4.0):
    """Sovrappone le frecce del flusso ottico sui frame, dopo aver sfocato il campo.
    N.B.: blur_sigma linked to blur_ksize (ksize ≈ 6 × sigma + 1  (arrotondato a dispari))"""

    # Sfoco il campo di flusso per smussare gli outlier
    fx = cv2.GaussianBlur(flow[...,0], (blur_ksize, blur_ksize), blur_sigma)
    fy = cv2.GaussianBlur(flow[...,1], (blur_ksize, blur_ksize), blur_sigma)

    h, w = image.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    # campiono dai campi blurati
    sampled_fx = fx[y, x]
    sampled_fy = fy[y, x]

    for (xi, yi, fxi, fyi) in zip(x, y, sampled_fx, sampled_fy):
        cv2.arrowedLine(image, (xi, yi),
                        (int(xi + fxi), int(yi + fyi)),
                        (0, 255, 0), 1, tipLength=0.2)
    return image

def save_plot_temporal_metrics(output_base, dish_well, metrics, start_frame):
    """Genera e salva un grafico con le metriche calcolate."""
    metric_colors = {
        'mean_magnitude': '#1f77b4',    # Blue
        'vorticity': '#2ca02c',         # Green
        'std_dev': '#d62728',           # Red
        'hybrid': '#ff7f0e',            # Orange
        'sum_mean_mag': '#9467bd'       # Purple
        }
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(16, 14))
    peak_config = {
        'distance': 12
        }

    for ax, (metric_name, values) in zip(axes, metrics.items()):
        # Adjust frame numbers (add initial cut frames)
        adjusted_frames = np.arange(len(values)) + start_frame

        # Plot main metric line
        ax.plot(adjusted_frames, values, 
                lw=1.8, 
                color=metric_colors[metric_name],
                label=f'{metric_name} trend')
        
        # Find and annotate peaks
        values = np.nan_to_num(values)  # Handle potential NaNs
        # Find peaks in both positive and negative directions
        peaks_pos, _ = find_peaks(values, **peak_config)
        peaks_neg, _ = find_peaks(-values, **peak_config)
        all_peaks = np.unique(np.concatenate((peaks_pos, peaks_neg)))

        if len(all_peaks) > 0:
            peak_abs = np.abs(values[all_peaks])
            top_indices = np.argsort(peak_abs)[-5:]  # Get indices of top 5 magnitudes
            top_peaks = all_peaks[top_indices]

            # Plot peaks with annotations
            for idx, peak in enumerate(top_peaks):
                frame_num = peak + start_frame
                value = values[peak]
                color = '#d62728' if value > 0 else '#1f77b4'  # Red/Blue
                marker = '^' if value > 0 else 'v'  # Triangle up/down
                ax.scatter(frame_num, value,
                           c=color, marker=marker,
                           zorder=5, s=80, edgecolor='black')
                
                # Annotate with frame number
                ax.text(frame_num, 
                        value*(1.1 if value > 0 else 0.9), 
                        f'Frame {frame_num}',
                        ha='center', va='bottom' if value > 0 else 'top',
                        rotation=45, fontsize=8,
                        bbox=dict(facecolor='white', 
                                  alpha=0.8, 
                                  edgecolor='lightgray',
                                  boxstyle='round,pad=0.2'))
        
        # Styling
        ax.set_title(f"{metric_name.upper()} Evolution", fontsize=12, pad=10)
        ax.set_xlabel("Frame Number (Original Timeline)", fontsize=10)
        ax.set_ylabel(metric_name.capitalize(), fontsize=10)
        ax.grid(alpha=0.2)
        ax.legend(loc='upper right', fontsize=8)

        # Set consistent x-axis limits
        ax.set_xlim(adjusted_frames[0]-5, adjusted_frames[-1]+5)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_base, f"{dish_well}_metrics.png"), 
               dpi=200, 
               bbox_inches='tight')
    plt.close()

def preprocess_frame(frame, use_clahe=True, clahe_clip=2.0, sigma_blur=1.5):
    """Fixed preprocessing with correct CLAHE application"""
    processed = frame.copy()

    # Ensure input is uint8 (original image format)
    if processed.dtype != np.uint8:
        processed = processed.astype(np.uint8)

    # CLAHE and Gaussian Blur (common steps)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8,8))
        processed = clahe.apply(processed)
    if sigma_blur > 0:
        k = int(6*sigma_blur + 1) | 1
        processed = cv2.GaussianBlur(processed, (k,k), sigma_blur)

    return processed

def calculate_vorticity(flow):
    # Calcola il gradiente del flusso
    grad = np.gradient(flow)
    
    # La vorticity è la differenza tra le derivate delle componenti del flusso
    vorticity = np.mean(grad[1][..., 1] - grad[0][..., 0])
    
    return vorticity


def calculate_vorticity_field(flow):
    # Calcola il gradiente del flusso
    grad = np.gradient(flow)

    # La vorticity è la differenza tra le derivate delle componenti del flusso
    vorticity_field = grad[1][..., 1] - grad[0][..., 0]
    
    return vorticity_field


def compute_optical_flowFarneback(prev_frame, current_frame,
                                  pyr_scale,
                                  levels,
                                  winSize,
                                  iterations,
                                  poly_n,
                                  poly_sigma,
                                  flags):
    
    """Enhanced Farneback optical flow with hybrid GPU/CPU processing"""
    
    # Check OpenCL availability once at start
    ocl_available = cv2.ocl.haveOpenCL()
    if ocl_available:
        # Enable OpenCL for all OpenCV operations
        cv2.ocl.setUseOpenCL(True)
        
        # Convert to UMat preserving original data
        prev_umat = cv2.UMat(prev_frame.astype(np.float32))
        curr_umat = cv2.UMat(current_frame.astype(np.float32))
    else:
        prev_umat = prev_frame.astype(np.float32)
        curr_umat = current_frame.astype(np.float32)

    try:
        # Compute flow with automatic GPU fallback
        flow = cv2.calcOpticalFlowFarneback(
            prev_umat, curr_umat, None,
            pyr_scale=pyr_scale,
            levels=levels,
            winsize=winSize,
            iterations=iterations,
            poly_n=poly_n,
            poly_sigma=poly_sigma,
            flags=flags
            )
        
        # If using OpenCL, convert UMat to numpy array
        if isinstance(flow, cv2.UMat):
            flow = flow.get()
            
        # GPU-accelerated polar conversion if available
        if ocl_available:
            mag_umat, ang_umat = cv2.cartToPolar(
                cv2.UMat(flow[...,0]), 
                cv2.UMat(flow[...,1]))
            magnitude = mag_umat.get()
            angle = ang_umat.get()
        else:
            magnitude, angle = cv2.cartToPolar(flow[...,0], flow[...,1])
            
    except cv2.error as e:
        print(f"Optical Flow Error: {e}")
        # Fallback to CPU-only processing
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame, current_frame, None,
            pyr_scale=pyr_scale,
            levels=levels,
            winsize=winSize,
            iterations=iterations,
            poly_n=poly_n,
            poly_sigma=poly_sigma,
            flags=flags
            )
        magnitude, angle = cv2.cartToPolar(flow[...,0], flow[...,1])

    # Converti l'angolo in gradi
    angle_degrees = np.rad2deg(angle) % 360

    return magnitude, angle_degrees, flow


def compute_optical_flowPyrLK(prev_frame, current_frame,
                              winSize,
                              maxLevelPyramid,
                              maxCorners, 
                              qualityLevel, 
                              minDistance, 
                              blockSize
                              ):

    # Convert frames to GPU Mats (if CUDA available)
    use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
    if use_cuda:
        gpu_prev = cv2.cuda_GpuMat(prev_frame)
        gpu_curr = cv2.cuda_GpuMat(current_frame)
    else:
        # Fallback to OpenCL via UMat
        prev_umat = cv2.UMat(prev_frame)
        curr_umat = cv2.UMat(current_frame)

    # Trovo i punti di interesse nel frame precedente
    # goodFeaturesToTrack() trova i punti di interesse nel prev_frame. 
    # Questi punti di interesse sono selezionati utilizzando l'algoritmo di Shi-Tomasi
    prev_pts = cv2.goodFeaturesToTrack(prev_frame, 
                                       maxCorners=maxCorners, 
                                       qualityLevel=qualityLevel, 
                                       minDistance=minDistance, 
                                       blockSize=blockSize,
                                       )
    
    """
    Parametri per il tracciamento dei punti di interesse:
    - Lucas-Kanade è l'algoritmo e nei parametri "maxLevel" si riferisce alla profondità della piramide
    - maxCorners è il numero massimo di punti di interesse
    - qualityLevel è qualità richiesta (tengo bassa)
    - minDistance è minima distanza tra due punti di interesse (ne viene mantenuto solo uno se sono più vicini di minDistance)
    - blockSize è dimensione block per algoritmo. 
    Se l'immagine contiene dettagli fini, è consigliabile utilizzare una finestra più piccola per individuare i punti chiave in aree più precise
    """

    if prev_pts is None:
        return np.zeros(0), np.zeros(0), np.zeros((0,2)), np.zeros((0,2))

    # cornerSubPix for refined feature localization
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    prev_pts = cv2.cornerSubPix(prev_frame, prev_pts, (10,10), (-1,-1), criteria)


    # GPU optical flow computation
    if use_cuda:
        # CUDA LK implementation
        lk = cv2.cuda.SparsePyrLKOpticalFlow_create(
            winSize=(winSize, winSize),
            maxLevel=maxLevelPyramid,
            numIters=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
        
        # Upload points to GPU
        gpu_prev_pts = cv2.cuda_GpuMat(prev_pts.astype(np.float32))
        
        # Calculate flow
        gpu_next_pts, status = lk.calc(
            gpu_prev, gpu_curr,
            gpu_prev_pts, None
            )
        
        # Download results
        next_pts = gpu_next_pts.download()
        status = status.download()
    else:
        # OpenCL acceleration
        lk_params = dict(
            winSize=(winSize, winSize),
            maxLevel=maxLevelPyramid,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
        
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_umat, curr_umat,
            prev_pts.astype(np.float32), None, **lk_params
            )
        next_pts = cv2.UMat.get(next_pts) if isinstance(next_pts, cv2.UMat) else next_pts
        status = cv2.UMat.get(status) if isinstance(status, cv2.UMat) else status

    # Filter valid points
    status = status.ravel().astype(bool)
    good_prev = prev_pts[status]
    good_next = next_pts[status]

    # Convert sparse to dense flow
    h, w = prev_frame.shape
    dense_flow = np.zeros((h, w, 2), dtype=np.float32)

    if len(good_prev) > 100:  # Only interpolate with sufficient points
        from scipy.interpolate import griddata
        x = np.linspace(0, w-1, w)
        y = np.linspace(0, h-1, h)
        xx, yy = np.meshgrid(x, y)

        # Reshape points to (N, 2)
        good_prev = good_prev.reshape(-1, 2)
        good_next = good_next.reshape(-1, 2)
        
        # Calculate displacements
        displacements = good_next - good_prev
        
        # Griddata interpolation
        dense_flow[...,0] = griddata(
            good_prev, 
            displacements[:,0],
            (xx, yy), 
            method='linear', 
            fill_value=0
        )
        dense_flow[...,1] = griddata(
            good_prev,
            displacements[:,1],
            (xx, yy),
            method='linear',
            fill_value=0
        )

    # Calcola la magnitudo e l'angolo del flusso ottico
    magnitude, angle = cv2.cartToPolar(dense_flow[..., 0], dense_flow[..., 1])

    # Converti l'angolo in gradi
    angle_degrees = np.rad2deg(angle) % 360

    return magnitude, angle_degrees, dense_flow, prev_pts
