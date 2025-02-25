import numpy as np
import cv2
import os

import sys
# Configurazione dei percorsi e dei parametri
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_01_OpticalFlow as conf
from config import utils as utils

def sort_files_by_slice_number(file_list):
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


def calculate_vorticity(flow):
    # Calcola il gradiente del flusso
    grad = np.gradient(flow)
    
    # La vorticity è la differenza tra le derivate delle componenti del flusso
    vorticity = np.mean(grad[1][..., 0] - grad[0][..., 1])
    
    return vorticity


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



def compute_optical_flowPyrLK(prev_frame, current_frame):
    # Parametri per il tracciamento dei punti di interesse
    # Lucas-Kanade è l'algoritmo e nei parametri "maxLevel" si riferisce alla profondità della piramide
    lk_params = dict(winSize=(conf.winSize, conf.winSize),
                     maxLevel=conf.maxLevelPyramid,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Trovo i punti di interesse nel frame precedente
    # goodFeaturesToTrack() trova i punti di interesse nel prev_frame. 
    # Questi punti di interesse sono selezionati utilizzando l'algoritmo di Shi-Tomasi
    prev_pts = cv2.goodFeaturesToTrack(prev_frame, maxCorners=conf.maxCorners, qualityLevel=conf.qualityLevel, minDistance=conf.minDistance, blockSize=conf.blockSize)
    # maxCorners è il numero massimo di punti di interesse
    # qualityLevel è qualità richiesta (tengo bassa)
    # minDistance è minima distanza tra due punti di interesse (ne viene mantenuto solo uno se sono più vicini di minDistance)
    # blockSize è dimensione block per algoritmo. 
    # Se l'immagine contiene dettagli fini, è consigliabile utilizzare una finestra più piccola per individuare i punti chiave in aree più precise

    # Calcolo il flusso ottico tra i frame
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, current_frame, prev_pts, None, **lk_params)

    # Filtro i punti di interesse e calcolo il flusso ottico
    good_prev = prev_pts[status == 1]
    good_next = next_pts[status == 1]
    flow = good_next - good_prev
    # calcolato il flusso ottico sottraendo le posizioni dei punti di interesse nel frame
    # corrente dalle posizioni nel frame precedente

    # Calcola la magnitudo e l'angolo del flusso ottico
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Converti l'angolo in gradi
    angle_degrees = np.rad2deg(angle) % 360

    return magnitude, angle_degrees, flow, prev_pts



