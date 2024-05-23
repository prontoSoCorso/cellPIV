import numpy as np


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



def calculate_vorticity(flow):
    # Calcola il gradiente del flusso
    grad = np.gradient(flow)
    
    # La vorticity è la differenza tra le derivate delle componenti del flusso
    vorticity = np.mean(grad[1][..., 0] - grad[0][..., 1])
    
    return vorticity



