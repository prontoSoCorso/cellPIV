import cv2
import numpy as np
import os

# Funzione per il calcolo del flusso ottico tra il frame precedente e quello attuale
# (QUI DA IMPLEMENTARE IL METODO CHE FANNO NEL CellPIV)
def compute_optical_flow(prev_frame, current_frame):
    # Parameters for Farneback Optical Flow
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
    # vorticity media  in ogni frame (1 valore per ogni frame che indica la media delle fasi delle coordinate polari (rotazione media))
    # media dell'inverso della deviazione standard delle direzioni dei vettori
    # hybrid: insieme di mean magnitude e st.dev (quello inverso medio calcolato prima. Non specifica come, guardo meglio)
    avg_magnitude = []
    max_magnitude = []
    total_displacement = 0
    directional_changes = 0
    speed_histogram = []
    trajectory_length = 0
    velocity_variance = []

    for frame_file in enumerate(frame_files[1:]):
        # Read the current frame
        current_frame = cv2.imread(os.path.join(folder_path, frame_file), cv2.IMREAD_GRAYSCALE)
        current_frame = cv2.resize(current_frame, target_size)

        # Compute optical flow
        magnitude, angle_degrees, flow = compute_optical_flow(prev_frame, current_frame)

        # Calculate metrics
        avg_magnitude.append(np.mean(magnitude))
        max_magnitude.append(np.max(magnitude))
        total_displacement += np.sum(magnitude)
        speed_histogram.extend(magnitude)
        trajectory_length += np.sum(magnitude)
        velocity_variance.append(np.var(magnitude))

        # Count directional changes
        directional_changes += np.sum(np.abs(np.diff(flow[..., 0])) > 0.5)

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


if __name__ == "__main__":
    for class_sample in ['blasto','no_blasto']:
        path_all_folders = "../Data"+class_sample
        errors = []
        
        for sample in os.listdir(path_all_folders):
            try:
                sample_path = os.path.join(path_all_folders, sample)
                sample_name = sample

                if 'no_blasto' in path_all_folders:
                    output_folder = "../Data/addImgsWithOpticalFlow/no_blasto/"+sample_name+"/"
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                else:
                    output_folder = "../Data/addImgsWithOpticalFlow/blasto/"+sample_name+"/"
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