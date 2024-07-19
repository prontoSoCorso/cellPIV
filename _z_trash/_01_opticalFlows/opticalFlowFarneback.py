import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import timeit
from scipy.signal import find_peaks
from myFunctions import calculate_vorticity, sort_files_by_slice_number


def compute_optical_flowFarneback(prev_frame, current_frame):
    """
        Calcola il flusso ottico tra due immagini usando algoritmo Farneback (usato da calcOpticalFlowFarneback).

        Parametri:
            prev_frame: La prima immagine.
            current_frame: La seconda immagine.

        Restituisce:
            - La magnitudo del flusso ottico.
            - L'angolo del flusso ottico in gradi.
            - Il campo vettoriale del flusso ottico.
    """

    # Parametri per Farneback Optical Flow
    flow_params = dict(
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    if (prev_frame.shape != current_frame.shape):
        print("==========================================================")
        print(prev_frame.shape)
        print(current_frame.shape)


    flow = cv2.calcOpticalFlowFarneback(prev_frame, current_frame, None, **flow_params)

    '''
    flow ha shape (500,500,2). La dimensione è la stessa dell'immagine e ha due canali che sono flusso
    verticale e orizzontale. La spiegazione dei canali è:
    Canale x (orizzontale): Questo canale contiene i valori del flusso ottico per la componente orizzontale. 
                            Per ogni pixel, il valore rappresenta la velocità di spostamento in direzione orizzontale tra il frame precedente e quello attuale. 
                            I valori positivi indicano uno spostamento verso destra, mentre i valori negativi indicano uno spostamento verso sinistra.
    Canale y (verticale): Questo canale contiene i valori del flusso ottico per la componente verticale. 
                            Per ogni pixel, il valore rappresenta la velocità di spostamento in direzione verticale tra il frame precedente e quello attuale. 
                            I valori positivi indicano uno spostamento verso il basso, mentre i valori negativi indicano uno spostamento verso l'alto.

    '''

    # Compute the magnitude and angle of the flow
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Convert the angle to degrees
    angle_degrees = np.rad2deg(angle) % 360

    return magnitude, angle_degrees, flow


def calculate_vorticity(flow):
    '''
    # Componenti
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]

    # Gradienti spaziali e vorticity
    grad_x = cv2.Sobel(flow_x, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(flow_y, cv2.CV_64F, 0, 1, ksize=5)
    vorticity = np.mean(grad_y - grad_x)
    '''

    # Calcola il gradiente del flusso
    grad = np.gradient(flow)
    
    # La vorticity è la differenza tra le derivate delle componenti del flusso
    vorticity = np.mean(grad[1][..., 0] - grad[0][..., 1])


    return vorticity


def overlay_arrows(frame, magnitude, angle_degrees, step=16):
    # Overlay arrows 
    for y in range(0, frame.shape[0], step):
        for x in range(0, frame.shape[1], step):
            # Get the flow at this point
            flow_magnitude = magnitude[y, x]
            flow_angle = angle_degrees[y, x]

            # Calculate the endpoint of the arrow
            endpoint = (int(x + flow_magnitude * np.cos(np.radians(flow_angle))),
                        int(y + flow_magnitude * np.sin(np.radians(flow_angle))))

            # Draw the arrow
            cv2.arrowedLine(frame, (x, y), endpoint, (255, 0, 0), 1)

    return frame


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


def process_frames(folder_path, output_folder):
    target_size = (500, 500) #DIPENDE DAL METODO USATO, qui non necessario, aumenterei solo efficienza

    # Get the list of frames in the folder
    frame_files = sort_files_by_slice_number(os.listdir(folder_path))

    # Intorno al frame 290 ho sempre picco causa cambio terreno (giorno 3). Taglio video a frame 288 per questo (3 giorni esatti)
    # Per i video S0675 a 261/262 ho un mini shift di immagini (senza apparente motivo), quindi trovo soluzione.
    # Soluzione 1 --> considero delle finestre temporali invece che solo singolo momento per trovare punti divisione cellulare
    # Soluzione 2 --> 
    frame_files = frame_files[:250]

    # Read the first frame
    prev_frame = cv2.imread(os.path.join(folder_path, frame_files[0]), cv2.IMREAD_GRAYSCALE)
    prev_frame = cv2.resize(prev_frame, target_size)

    # Metrics initialization (QUI DA CAMBIARE)

    # media della magnitudine in ogni frame (1 valore per frame)
    # media della magnitudine tra un numero di frame specifico in avanti nel tempo (?)
    # vorticity media in ogni frame (1 valore per ogni frame che indica la media delle fasi delle coordinate polari (rotazione media))
    # media dell'inverso della deviazione standard delle direzioni dei vettori
    # hybrid: insieme di mean magnitude e st.dev (quello inverso medio calcolato prima. Non specifica come, guardo meglio)

    # MEAN MAG, VORT, ST DEV
    mean_magnitude = []
    vorticity = []
    std_dev = []
    hybrid = []

    for i, frame_file in enumerate(frame_files[1:]):

        # Read the current frame
        current_frame = cv2.imread(os.path.join(folder_path, frame_file), cv2.IMREAD_GRAYSCALE)
        current_frame = cv2.resize(current_frame, target_size)

        # Compute optical flow
        magnitude, angle_degrees, flow = compute_optical_flowFarneback(prev_frame, current_frame)

        # Overlay arrows on the current frame
        frame_with_arrows = overlay_arrows(prev_frame.copy(), magnitude, angle_degrees)

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

    print(f"Processed frames saved to {output_folder}")

    #Find 5 peaks for each metric
    peaks_mag, _ = find_peaks(mean_magnitude, distance=10)  # Trova i picchi con una distanza minima di n frame tra loro
    tmp_vort = [abs(x) for x in vorticity]
    peaks_vort, _ = find_peaks(tmp_vort, distance=10)      
    peaks_std_dev, _ = find_peaks(std_dev, distance=10)
    peaks_hybrid, _ = find_peaks(hybrid, distance=10)
    peaks_sum_mean_mag, _ = find_peaks(sum_mean_mag, distance=10)


    # Trovo i valori e gli indici dei 5 picchi più alti
    mean_magnitude = np.array(mean_magnitude).astype(float)     # Devo convertirlo perché è una lista in partenza
    vorticity = np.array(vorticity).astype(float)     # Devo convertirlo perché è una lista in partenza
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
    plt.show()

    # Call saveFig function to save the plot
    saveFig(folder_path, output_folder)


    '''
    plt.figure()
    plt.plot(time_steps, mean_magnitude, label='Mean Magnitude')
    plt.plot(time_steps, vorticity, label='Vorticity')

    plt.scatter(top_5_peaks_indices_mag, top_5_peaks_values_mag, color='blue', marker='o', facecolors='none', label='Peaks (Magnitude)')
    plt.scatter(top_5_peaks_indices_vort, top_5_peaks_values_vort, color='red', marker='o', facecolors='none', label='Peaks (Vorticity)')

    # Aggiungi testo con gli indici vicino ai pallini
    for i, (index_mag, index_vort) in enumerate(zip(top_5_peaks_indices_mag, top_5_peaks_indices_vort)):
        plt.text(index_mag, top_5_peaks_values_mag[i], str(index_mag), fontsize=9, va='bottom', ha='right', color='blue')
        plt.text(index_vort, top_5_peaks_values_vort[i], str(index_vort), fontsize=9, va='bottom', ha='left', color='red')

    plt.xlabel('Frame Index')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.title("Temporal Analysis of Optical Flow's magnitude - Farneback")

    # Call saveFig function to save the plot
    saveFig(folder_path, output_folder)

    '''

    


    '''
    # Additional Metrics
    print(f"Average Magnitude: {np.mean(avg_magnitude)}")
    print(f"Maximum Magnitude: {np.max(max_magnitude)}")
    print(f"Total Displacement: {total_displacement}")
    print(f"Directional Changes: {directional_changes}")

    # Speed Histogram
 
    plt.hist(speed_histogram, bins=20, alpha=0.7, histtype='step')
    plt.title('Speed Histogram')
    plt.xlabel('Magnitude')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(metrics_video_folder, 'speed_histogram.png'))
    plt.close()

    # Trajectory Length
    print(f"Trajectory Length: {trajectory_length}")

    # Velocity Variance
    print(f"Velocity Variance: {np.mean(velocity_variance)}")

    # Frequency of Directional Changes
    print(f"Frequency of Directional Changes: {directional_changes}")

    # Temporal Analysis (Plotting metrics over time)
    time_steps = np.arange(len(frame_files) - 1)
    plt.plot(time_steps, avg_magnitude, label='Average Magnitude')
    plt.plot(time_steps, max_magnitude, label='Maximum Magnitude')
    plt.plot(time_steps, velocity_variance, label='Velocity Variance')
    plt.xlabel('Frame Index')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.title('Temporal Analysis of Optical Flow Metrics')
    plt.savefig(os.path.join(metrics_video_folder, 'temporal_analysis.png'))
    ''' 



def saveFig(folder_path, output_folder):
    # Find the index of the substring 'D20'
    index_d20 = folder_path.find('D20')
    # Extract the substring starting from the index of 'D20'
    substring_after_d20 = folder_path[index_d20:]

    # Trova l'indice della sottostringa specificata da method_optical_flow
    index_method = output_folder.find(method_optical_flow)

    # Estrai la parte della stringa fino all'indice trovato
    partial_path_till_method_optical = output_folder[:index_method + len(method_optical_flow)]

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
    # Parametri da impostare in base al metodo di flusso ottico, utile solo per cartelle
    global method_optical_flow 
    method_optical_flow = "Farneback"

    for class_sample in ['blasto','no_blasto']:
        path_all_folders = "C:/Users/loren/Documents/Data/BlastoData/"+class_sample
        errors = []

        for sample in os.listdir(path_all_folders):
            try:
                sample_path = path_all_folders + "/" + sample
                sample_name = sample    #sample_name è il nome della cartella

                if 'no_blasto' in path_all_folders:
                    output_folder = "C:/Users/loren/Documents/Data/BlastoData/opticalFlowFrames/" + method_optical_flow + "/no_blasto/" + sample_name + "/"
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)

                else:
                    output_folder = "C:/Users/loren/Documents/Data/BlastoData/opticalFlowFrames/" + method_optical_flow + "/blasto/"    + sample_name + "/"
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)

                process_frames(sample_path,output_folder)

            except Exception as e:
                print('-------------------')
                print("Error in sample: ", sample_name)
                print(e)
                errors.append(sample_name)
                print('-------------------')
                continue
                
        print('Errors: ', errors)





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

    # Calcola il flusso ottico tra i due frame
    magnitude, angle_degrees, flow = compute_optical_flowFarneback(prev_frame, current_frame)

    # Sovrappongo il flusso ottico al secondo frame
    overlaid_frame = overlay_arrows(current_frame.copy(), magnitude, angle_degrees)

    # Salvataggio del frame con le frecce 
    output_folder = "C:/Users/loren/Documents/Data/BlastoData/opticalFlowFrames/frame_con_flusso_ottico_noResize.jpg"
    cv2.imwrite(output_folder, overlaid_frame)



# Misura il tempo di esecuzione della funzione main()
main()

#execution_time = timeit.timeit(main, number=1)

#print("Tempo impiegato:", execution_time, "secondi")

