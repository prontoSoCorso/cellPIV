import os
import pickle
import cv2
from matplotlib import pyplot as plt
import numpy as np
import timeit
from scipy.signal import find_peaks
from myFunctions import calculate_vorticity, sort_files_by_slice_number

import sys
# Configurazione dei percorsi e dei parametri
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_01_OpticalFlow as conf
from config import user_paths as myPaths



def compute_optical_flowPyrLK(prev_frame, current_frame):
    # Parametri per il tracciamento dei punti di interesse
    # Lucas-Kanade è l'algoritmo e nei parametri "maxLevel" si riferisce alla profondità della piramide
    lk_params = dict(winSize=(10, 10),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Trovo i punti di interesse nel frame precedente
    # goodFeaturesToTrack() trova i punti di interesse nel prev_frame. 
    # Questi punti di interesse sono selezionati utilizzando l'algoritmo di Shi-Tomasi
    prev_pts = cv2.goodFeaturesToTrack(prev_frame, maxCorners=300, qualityLevel=0.3, minDistance=10, blockSize=7)
    # maxCorners è il numero massimo di punti di interesse
    # qualityLevel è qualità richiesta (tengo bassa)
    # minDistance è minima distanza tra due punti di interesse (ne viene mantenuto solo uno se sono più vicini di minDistance)
    # blockSize è dimensione block per algoritmo. 
    #       Se l'immagine contiene dettagli fini, è consigliabile utilizzare una finestra più piccola per individuare i punti chiave in aree più precise

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




def process_frames(folder_path, output_folder):
    target_size = (conf.img_size, conf.img_size)    # Impongo dimensione altrimenti dà problemi

    # Get the list of frames in the folder
    frame_files = sort_files_by_slice_number(os.listdir(folder_path))

    # Se non ho visto male, al frame 293 ho sempre picco. Probabilmente cambio terreno (giorno 3 se frame ogni 15 min)
    # Invece a 261/262 ho un mini shift di immagini (senza apparente motivo), quindi taglio ancora prima
    frame_files = frame_files[:conf.num_frames]

    # Read the first frame
    prev_frame = cv2.imread(os.path.join(folder_path, frame_files[0]), cv2.IMREAD_GRAYSCALE)
    prev_frame = cv2.resize(prev_frame, target_size)

    # METRICHE
    # media della magnitudine in ogni frame (1 valore per frame)
    # media della magnitudine tra un numero di frame specifico in avanti nel tempo (?)
    # vorticity media in ogni frame (1 valore per ogni frame che indica la media delle fasi delle coordinate polari (rotazione media))
    # media dell'inverso della deviazione standard delle direzioni dei vettori
    # hybrid: insieme di mean magnitude e st.dev (quello inverso medio calcolato prima. Non specifica come)

    mean_magnitude = []
    vorticity = []
    std_dev = []
    hybrid = []

    for i, frame_file in enumerate(frame_files[1:]):

        # Read the current frame
        current_frame = cv2.imread(os.path.join(folder_path, frame_file), cv2.IMREAD_GRAYSCALE)
        current_frame = cv2.resize(current_frame, target_size)

        # Compute optical flow
        magnitude, angle_degrees, flow, prev_pts = compute_optical_flowPyrLK(prev_frame, current_frame)

        # Overlay arrows on the current frame
        frame_with_arrows = overlay_arrows(prev_frame.copy(), magnitude, angle_degrees, prev_pts)

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

        # Save the processed frame
        output_path = os.path.join(output_folder, frame_files[i])
        cv2.imwrite(output_path, frame_with_arrows)

        # Update the previous frame for the next iteration
        prev_frame = current_frame

    # Compute sum mean mag
    sum_mean_mag = []
    num_forward_frame = 4

    for frame in range(0, len(mean_magnitude)-num_forward_frame+1):
        sum_mean_mag.append(np.mean(mean_magnitude[frame:frame+num_forward_frame]))
    
    # Pad sum_mean_mag with zeros to match mean_magnitude length
    padding_length = len(mean_magnitude) - len(sum_mean_mag)
    sum_mean_mag = np.pad(sum_mean_mag, ((0, padding_length)), 'constant', constant_values=(0))

    

    if conf.save_images:
        process_and_save_metrics(folder_path, output_folder, frame_files, mean_magnitude, vorticity, std_dev, hybrid, sum_mean_mag)
        print(f"Processed frames saved to {output_folder}")

    else:
        mean_magnitude = np.array(mean_magnitude).astype(float)     # Devo convertirlo perché è una lista in partenza
        vorticity = np.array(vorticity).astype(float)
        std_dev = np.array(std_dev).astype(float)    
        hybrid = np.array(hybrid).astype(float)
        sum_mean_mag = np.array(sum_mean_mag).astype(float)
        print(f"Processed frames {output_folder} saved successfully")   # In questo caso "output folder" sarebbe il sample_name

    return mean_magnitude, vorticity, hybrid, sum_mean_mag






def process_and_save_metrics(folder_path, output_folder, frame_files, mean_magnitude, vorticity, std_dev, hybrid, sum_mean_mag):
    #Find 5 peaks for each metric
    peaks_mag, _ = find_peaks(mean_magnitude, distance=10)  # Trova i picchi con una distanza minima di n frame tra loro
    tmp_vort = [abs(x) for x in vorticity]
    peaks_vort, _ = find_peaks(tmp_vort, distance=10)      
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
    time_steps = np.arange(len(frame_files) - 1)

    fig, axes = plt.subplots(5, 1, figsize=(12, 10))

    data_labels = [
        (mean_magnitude, 'Mean Magnitude', 'Magnitude', top_5_peaks_indices_mag, top_5_peaks_values_mag, 'blue', 'Magnitude'),
        (vorticity, 'Vorticity', 'Vorticity', top_5_peaks_indices_vort, top_5_peaks_values_vort, 'red', 'Vorticity'),
        (std_dev, 'Standard Deviation', 'Standard Deviation', top_5_peaks_indices_std_dev, top_5_peaks_values_std_dev, 'purple', 'Standard Deviation'),
        (hybrid, 'Hybrid', 'Hybrid', top_5_peaks_indices_hybrid, top_5_peaks_values_hybrid, 'orange', 'Hybrid'),
        (sum_mean_mag, 'Sum Mean Magnitude', 'Sum Mean Magnitude', top_5_peaks_indices_sum_mean_mag, top_5_peaks_values_sum_mean_mag, 'green', 'Sum Mean Magnitude')
    ]

    # Plot the data in each subplot
    for i, (data, label, ylabel, peak_indices, peak_values, peak_color, peak_label) in enumerate(data_labels):
        axes[i].plot(time_steps, data, label=label, color=peak_color)
        axes[i].scatter(peak_indices, peak_values, color=peak_color, marker='o', facecolors='none', label=f'Peaks')
        for j, index_peak in enumerate(peak_indices):
            axes[i].text(index_peak, peak_values[j], str(index_peak), fontsize=9, va='bottom', ha='right' if i < 2 else 'left', color=peak_color)
        axes[i].set_xlabel('Frame Index')
        axes[i].set_ylabel(ylabel)
        axes[i].legend()
        axes[i].set_title(f"Temporal Analysis of {label}")

    plt.tight_layout()

    # Call saveFig function to save the plot
    saveFig(folder_path, output_folder)




def saveFig(folder_path, output_folder):
    # Find the index of the substring 'D20'
    index_d20 = folder_path.find('D20')
    # Extract the substring starting from the index of 'D20'
    substring_after_d20 = folder_path[index_d20:]

    # Trova l'indice della sottostringa specificata da method_optical_flow
    index_method = output_folder.find(conf.method_optical_flow)

    # Estrai la parte della stringa fino all'indice trovato
    partial_path_till_method_optical = output_folder[:index_method + len(conf.method_optical_flow)]

    if 'no_blasto' in folder_path:
        output_folder_metrics = partial_path_till_method_optical + "/metrics/no_blasto/"+substring_after_d20+"/"
        if not os.path.exists(output_folder_metrics):
            os.makedirs(output_folder_metrics)

    else:
        output_folder_metrics = partial_path_till_method_optical + "/metrics/blasto/"   +substring_after_d20+"/"
        if not os.path.exists(output_folder_metrics):
            os.makedirs(output_folder_metrics)

    plt.savefig(output_folder_metrics + 'temporal_analysis.png')





def main():
    # Success/Error
    n_video_error = 0
    n_video_success = 0

    # Initialize matrices to store metrics for all videos
    mean_mag_mat = []
    vorticity_mat = []
    hybrid_mat = []
    sum_mean_mag_mat = []

    for class_sample in ['blasto','no_blasto']:
        path_all_folders = myPaths.path_BlastoData + class_sample
        errors = []

        # Set the value to add based on the class_sample
        value_to_add = np.array([0]) if 'no_blasto' in class_sample else np.array([1])

        for sample in os.listdir(path_all_folders):
            try:
                sample_path = path_all_folders + "/" + sample
                sample_name = sample    #sample_name è il nome della cartella


                if conf.save_images:
                    # Voglio salvare le immagini delle metriche di flusso ottico per ogni paziente
                    if 'no_blasto' in path_all_folders:
                        output_folder = myPaths.path_BlastoData + "opticalFlowFrames/" + conf.method_optical_flow + "/no_blasto/" + sample_name + "/"
                        if not os.path.exists(output_folder):
                            os.makedirs(output_folder)

                    else:
                        output_folder = myPaths.path_BlastoData + "opticalFlowFrames/" + conf.method_optical_flow + "/blasto/"    + sample_name + "/"
                        if not os.path.exists(output_folder):
                            os.makedirs(output_folder)

                    mean_magnitude, vorticity, hybrid, sum_mean_mag = process_frames(sample_path,output_folder)


                else:
                    # Non voglio salvare le immagini delle metriche di flusso ottico per ogni paziente
                    mean_magnitude, vorticity, hybrid, sum_mean_mag = process_frames(sample_path, sample_name)

                # Append new patient data to matrices
                mean_mag_mat.append(np.concatenate([mean_magnitude, value_to_add]))
                vorticity_mat.append(np.concatenate([vorticity, value_to_add]))
                hybrid_mat.append(np.concatenate([hybrid, value_to_add]))
                sum_mean_mag_mat.append(np.concatenate([sum_mean_mag[:-3], value_to_add]))
            
                n_video_success += 1

            except Exception as e:
                n_video_error += 1
                print('-------------------')
                print("Error in sample: ", sample_name)
                print(e)
                errors.append(sample_name)
                print('-------------------')
                continue
                
        print('Errors: ', errors)


    # Stampo quanti frame con successo e quanti errori
    print('===================')
    print('===================')
    print(f"Sono state elaborate con successo {n_video_success} serie temporali")
    print(f"Non sono state elaborate le serie temporali di {n_video_error} video")

    # Ottengo il percorso della cartella dello script corrente
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Salvataggio della lista come file utilizzando pickle nella cartella corrente
    with open(os.path.join(current_directory, 'mean_mag_mat.pkl'), 'wb') as mm:
        pickle.dump(mean_mag_mat, mm)

    with open(os.path.join(current_directory, 'vorticity_mat.pkl'), 'wb') as v:
        pickle.dump(vorticity_mat, v)

    with open(os.path.join(current_directory, 'hybrid_mat.pkl'), 'wb') as h:
        pickle.dump(hybrid_mat, h)

    with open(os.path.join(current_directory, 'sum_mean_mag_mat.pkl'), 'wb') as smm:
        pickle.dump(sum_mean_mag_mat, smm)



# Se non voglio misurare tempo
# main()

# Misura il tempo di esecuzione della funzione main()
execution_time = timeit.timeit(main, number=1)
print("Tempo impiegato:", execution_time, "secondi")













'''
def main_two_frames():
    # Percorso delle immagini di input
    prev_image_path = "C:/Users/loren/Documents/Data/BlastoData/blasto/D2013.02.19_S0675_I141_1/D2013.02.19_S0675_I141_1_12_0_41324.840721423614.jpg"
    current_image_path = "C:/Users/loren/Documents/Data/BlastoData/blasto/D2013.02.19_S0675_I141_1/D2013.02.19_S0675_I141_1_13_0_41324.85114539352.jpg"

    # Controllo del caricamento delle immagini
    try:
        prev_frame = cv2.imread(prev_image_path, cv2.IMREAD_GRAYSCALE)
        current_frame = cv2.imread(current_image_path, cv2.IMREAD_GRAYSCALE)
    except FileNotFoundError as e:
        print(f"Errore: File non trovato - {e}")
        exit()

    # Calcolo il flusso ottico tra i due frame
    magnitude, angle_degrees, flow, prev_pts = compute_optical_flowPyrLK(prev_frame, current_frame)

    # Sovrappongo il flusso ottico al secondo frame
    overlaid_frame = overlay_arrows(current_frame.copy(), magnitude, angle_degrees, prev_pts)

    # Salvataggio del frame con le frecce
    output_folder = "C:/Users/loren/Documents/Data/BlastoData/opticalFlowFrames/frame_con_flusso_otticoPyrLK_3.jpg"
    cv2.imwrite(output_folder, overlaid_frame)
'''