''' Configuration file for the project'''

import torch

giovanna = True

class user_paths:
    #Per computer fisso
    if giovanna:
        path_BlastoData = "/home/giovanna/Documents/Data/BlastoData/"
    
    else:
    #Per computer portatile
        path_BlastoData = "C:/Users/loren/Documents/Data/BlastoData/"


class utils:
    # Dim
    img_size=500
    num_frames=288
    
    # Seed
    seed = 2024


class Config_00_preprocessing:
    path_old_excel          = user_paths.path_BlastoData + "BlastoLabels.xlsx"
    path_single_csv         = user_paths.path_BlastoData + "BlastoLabels_singleFile.csv"
    path_singleWithID_csv   = user_paths.path_BlastoData + "BlastoLabels_singleFileWithID.csv"
    path_double_dish_excel  = user_paths.path_BlastoData + "pz con doppia dish.xlsx"



class Config_01_OpticalFlow:
    # Paths
    project_name = 'BlastoClass_y13-18_3days_288frames_optflow_LK'
    method_optical_flow = "LucasKanade"

    # LK parameters
    winSize = 10
    maxLevelPyramid = 3
    maxCorners = 300
    qualityLevel = 0.3
    minDistance = 10
    blockSize = 7

    # Var
    save_images = 0
    num_forward_frame = 4   # Numero di frame per sum_mean_mag
    


class Config_02_temporalData:
    #Paths
    csv_file_path = Config_00_preprocessing.path_singleWithID_csv
    output_csv_file_path            = user_paths.path_BlastoData + "FinalBlastoLabels.csv"
    output_csvNormalized_file_path  = user_paths.path_BlastoData + "Normalized_Final_BlastoLabels.csv"

    # Data
    temporalDataType = "sum_mean_mag_dict"



class Config_02_train:
    project_name = 'BlastoClass_y13-18_3days_288frames_optflow_LK'
    if giovanna:
        data_path = '/home/giovanna/Desktop/Lorenzo/Tesi Magistrale/cellPIV/_01_opticalFlows'
    else:
        data_path = 'C:/Users/loren/OneDrive - Università di Pavia/Magistrale - Sanità Digitale/Tesi Magistrale/cellPIV/_01_opticalFlows'
    
    keyAPIpath = "C:/Users/loren/Documents/keyAPIwandb.txt"
    local_dir = "C:/Utenti/loren/Documents/rayTuneResults"
    
    model_name = 'LSTM'
    dataset = "Blasto"
    seed = 2024
    perc_train = 0.7
    perc_val = 0.2

    num_epochs = 20
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




'''
class Config_02_Model_transf:
    device = torch.device("mps")
    max_len=5000 # max time series sequence length 
    n_head = 4 # number of attention head
    n_layer = 2 # number of encoder layer
    drop_prob = 0.1
    d_model = 200 # number of dimension ( for positional embedding)
    ffn_hidden = 512 # size of hidden layer before classification 
    feature = 1 # for univariate time series (1d), it must be adjusted for 1. 
    model =  Transformer(  d_model=d_model, details=True, n_head=n_head, max_len=max_len, seq_len=sequence_len, ffn_hidden=ffn_hidden, n_layers=n_layer, drop_prob=drop_prob, device=device)

    batch_size = 7

    summary(model, input_size=(batch_size,sequence_len,feature) , device=device)

    print(model)

'''




