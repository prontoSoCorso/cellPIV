import os
import sys
import numpy as np
import torch
from signal_grad_cam import TorchCamBuilder
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# add project root to path
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = current_dir
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_03_train as conf
from _03_train._b_LSTMFCN import LSTMFCN
import _04_test._testFunctions as _testFunctions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_models_path = conf.output_model_base_dir
day = 1

# main
def main():
    # Load test data
    df_test = _testFunctions.load_test_data(day)
    temporal_columns = [col for col in df_test.columns if col.startswith('value_')]
    X = df_test[temporal_columns].values
    y_true = df_test['BLASTO NY'].astype(int).values

    # Load model
    model_path = os.path.join(base_models_path, f"best_lstmfcn_model_{day}Days.pth")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = LSTMFCN(
        lstm_size=checkpoint['params']['lstm_size'],
        filter_sizes=tuple(map(int, checkpoint['params']['filter_sizes'].split(','))),
        kernel_sizes=tuple(map(int, checkpoint['params']['kernel_sizes'].split(','))),
        dropout=checkpoint['params']['dropout'],
        num_layers=checkpoint['params']['num_layers']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    threshold = checkpoint.get('best_threshold', 0.5)
    model.eval()

    # Prepare Data
    # convert to a list of numpy arrays, each of shape (channels, time)
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (N, T, 1)
    X_np = X_tensor.numpy().squeeze(-1)                           # (N, T)
    data_list = [x[np.newaxis, :] for x in X_np]                  # list of (1, T)
    data_labels = y_true.tolist()                                 # list of ints
    data_names  = df_test["dish_well"].tolist()                   # list of dish_well names

    # Subsampling
    specific_video = "D2013.03.09_S0695_I141_1"
    n_sample = 3
    
    if specific_video:
        try:
            index = data_names.index(specific_video)
            data_list   = [data_list[index]]    # Wrap in a list to keep it a list
            data_labels = [data_labels[index]]
            data_names  = [data_names[index]]
        except ValueError:                      # Handle the case where the specific video is not found
            print(f"\n==============================")
            print(f"Warning: '{specific_video}' not found in data_names.")
            print(f"==============================\n")
            exit()
    else:
        data_list   = data_list[:n_sample]
        data_labels = data_labels[:n_sample]
        data_names  = data_names[:n_sample]

    # Output dirs
    results_dir = os.path.join(current_dir, "GRADCAM_outputs")
    os.makedirs(results_dir, exist_ok=True)

    # Set SignalGrad-CAM variables
    class_names = ["no_blasto", "blasto"]
    target_classes = [0, 1]
    explainer_types = ["Grad-CAM", "HiResCAM"]
    target_layers_names   = ["conv3.0"]

    # Define the CAM builder
    cam_builder = TorchCamBuilder(model=model, class_names=class_names, time_axs=1)
    
    cams, predicted_probs, bar_ranges = cam_builder.get_cam(
        data_list           = data_list, 
        data_labels         = data_labels,
        target_classes      = target_classes,
        explainer_types     = explainer_types,
        target_layers       = target_layers_names, 
        softmax_final       = False,
        data_names          = data_names, 
        results_dir_path    = results_dir,
        aspect_factor       = 18
        )

    # Explain different input channels
    comparison_algorithm = "Grad-CAM"
    cam_builder.single_channel_output_display(
        data_list           = data_list,
        data_labels         = data_labels,
        predicted_probs_dict= predicted_probs,
        cams_dict           = cams,
        explainer_types     = comparison_algorithm,
        target_classes      = target_classes,
        target_layers       = target_layers_names,
        data_names          = data_names,
        fig_size            = (6, 4),
        grid_instructions   = (1, 1), 
        bar_ranges_dict     = bar_ranges,
        results_dir_path    = results_dir,
        dt                  = 10,
        line_width          = 0.5, 
        marker_width        = 30, 
        axes_names          = ("Time Steps", "Optical Flow (SMM value)")
        )
    
if __name__=="__main__":
    main()
