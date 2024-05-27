''' Configuration file for the project'''

import torch

class user_paths:
    # Per computer fisso
    path_BlastoData = "/home/giovanna/Documents/Data/BlastoData/"
    
    #Per computer portatile
    #path_BlastoData = "C:/Users/loren/Documents/Data/BlastoData/"


class Config_01_OpticalFlow:
    # Paths
    project_name = 'BlastoClass_y13-18_3days_288frames_optflow_LK'
    method_optical_flow = "LucasKanade"

    # Dim
    seed = 2024
    img_size=500
    num_frames=288

    # Var
    save_images = 0



class Config_02_Model:
    project_name = 'BlastoClass_y13-18_3days_288frames_optflow_LK'
    #ata_path = 'C:/Users/loren/OneDrive - Università di Pavia/Magistrale - Sanità Digitale/Tesi Magistrale/cellPIV/_01_opticalFlows'
    data_path = '/home/giovanna/Desktop/Lorenzo/Tesi Magistrale/cellPIV/_01_opticalFlows'
    model_name = 'LSTM'
    dataset = "Blasto"
    seed = 2024

    num_epochs = 100
    batch_size = 16                  # numero di sequenze prese
    learning_rate = 0.0005
    pos_weight = torch.tensor(1)
    img_size=500
    num_classes=2

    # Parametri LSTM
    hidden_size = 64                    # Dimensione della cella nascosta
    num_layers = 5                      # Numero di layer LSTM
    output_size = 1                     # Dimensione dell'output
    bidirectional = True               # Imposta a True se la rete è bidirezionale
    dropout_prob = 0.2                  # Dimensione dropout

    optimizer_type = "RMSprop"             # Tipo optimizer utilizzato

    exp_name = dataset + "," + model_name + "," + str(num_epochs) + "," + str(batch_size) + "," + str(learning_rate) + "," + optimizer_type + "," + str(bidirectional)




