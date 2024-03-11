# Importo librerie necessarie
import cv2
import numpy as np
import os
from scipy.ndimage import fourier_shift


def compute_optical_flow(prev_frame, current_frame):
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

    flow = cv2.calcOpticalFlowFarneback(prev_frame, current_frame, None, **flow_params)

    # Compute the magnitude and angle of the flow
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Convert the angle to degrees
    angle_degrees = np.rad2deg(angle) % 360

    return magnitude, angle_degrees, flow


def compute_optical_flow_matpiv(prev_frame, current_frame):
    """
        Calcola il flusso ottico tra due immagini usando MatPIV.

        Parametri:
            prev_frame: La prima immagine.
            current_frame: La seconda immagine.

        Restituisce:
            - La magnitudo del flusso ottico.
            - L'angolo del flusso ottico in gradi.
            - Il campo vettoriale del flusso ottico.
    """

    # Definire i parametri di MatPIV
    interrogation_region_size = 15
    overlap = 0.5
    search_region_size = interrogation_region_size * 2

    # Calcolare la cross-correlazione
    cross_correlation_distributions = fourier_shift(current_frame, -np.fft.fftshift(prev_frame))

    # Trovare i picchi delle distribuzioni di cross-correlazione
    displacements = np.argmax(cross_correlation_distributions, axis=2)

    # Calcolare il campo vettoriale della velocità
    velocity_field = displacements / search_region_size

    # Calcolare la magnitudo e l'angolo del flusso ottico
    magnitude, angle = cv2.cartToPolar(velocity_field[..., 0], velocity_field[..., 1])

    # Convertire l'angolo in gradi
    angle_degrees = np.rad2deg(angle) % 360

    return magnitude, angle_degrees, velocity_field




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

    # Use the sorted function with the custom key function
    sorted_files = sorted(file_list, key=get_slice_number)

    return sorted_files

def process_frames(folder_path, output_folder):
    target_size = (224, 224) #DIPENDE DAL METODO USATO, POTREBBE NON ESSERE NECESSARIO

    # Get the list of frames in the folder
    frame_files = sort_files_by_slice_number(os.listdir(folder_path))
    
    # Read the first frame
    prev_frame = cv2.imread(os.path.join(folder_path, frame_files[0]), cv2.IMREAD_GRAYSCALE)
    prev_frame = cv2.resize(prev_frame, target_size)
    
    # Metrics initialization (QUI DA CAMBIARE)

    # media della magnitudine in ogni frame (1 valore per frame)
    # media della magnitudine tra un numero di frame specifico in avanti nel tempo (?)
    # vorticity media in ogni frame (1 valore per ogni frame che indica la media delle fasi delle coordinate polari (rotazione media))
    # media dell'inverso della deviazione standard delle direzioni dei vettori
    # hybrid: insieme di mean magnitude e st.dev (quello inverso medio calcolato prima. Non specifica come, guardo meglio)
    mean_magnitude = []



    for frame_file in enumerate(frame_files[1:]):
        # Read the current frame
        current_frame = cv2.imread(os.path.join(folder_path, frame_file), cv2.IMREAD_GRAYSCALE)
        current_frame = cv2.resize(current_frame, target_size)

        # Compute optical flow
        magnitude, angle_degrees, flow = compute_optical_flow(prev_frame, current_frame)

        # Calculate metrics
        mean_magnitude.append(np.mean(magnitude))

        # Overlay arrows on the current frame
        frame_with_arrows = overlay_arrows(current_frame.copy(), magnitude, angle_degrees)

        # Save the processed frame
        output_path = os.path.join(output_folder, frame_file)
        cv2.imwrite(output_path, frame_with_arrows)

        # Update the previous frame for the next iteration
        prev_frame = current_frame

    print(f"Processed frames saved to {output_folder}")

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



path = "C:/Users/loren/Documents/Data/BlastoData/blasto/D2013.02.19_S0675_I141_1"
errors = []

for sample in os.listdir(path):
    try:
        sample_path = os.path.join(path, sample)
        sample_name = sample

        output_folder = "C:/Users/loren/Documents/Data/BlastoData/opticalFlowFrames/blasto/D2013.02.19_S0675_I141_1"+sample_name+"/"
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



'''
for class_sample in ['blasto','no_blasto']:
    path_all_folders = "C:/Users/loren/Documents/Data/BlastoData/"+class_sample
    errors = []
    
    for sample in os.listdir(path_all_folders):
        try:
            sample_path = os.path.join(path_all_folders, sample)
            sample_name = sample

            if 'no_blasto' in path_all_folders:
                output_folder = "C:/Users/loren/Documents/Data/BlastoData/opticalFlowFrames/no_blasto/"+sample_name+"/"
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
            else:
                output_folder = "C:/Users/loren/Documents/Data/BlastoData/opticalFlowFrames/blasto/"+sample_name+"/"
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

'''