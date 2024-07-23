''' Configuration file for the project'''

import torch
import random
import numpy as np
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
    num_classes=2

    # Seed everything
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
    csv_file_path                   = Config_00_preprocessing.path_singleWithID_csv
    output_csv_file_path            = user_paths.path_BlastoData + "FinalBlastoLabels.csv"
    output_csvNormalized_file_path  = user_paths.path_BlastoData + "Normalized_Final_BlastoLabels.csv"

    # Data
    temporalDataType = "sum_mean_mag_dict"



class Config_03_train_rocket:
    project_name = 'BlastoClass_y13-14_3days_288frames_optflow_LK'
    data_path = Config_02_temporalData.output_csvNormalized_file_path
    keyAPIpath = "C:/Users/loren/Documents/keyAPIwandb.txt"
    
    model_name = 'Rocket'
    dataset = "Blasto"
    num_kernels = 100000
    perc_train = 0.8
    perc_test = 0.2
    img_size = utils.img_size
    num_classes = utils.num_classes

    exp_name = dataset + "," + model_name + "," + str(num_kernels)

    # Seed
    seed = utils.seed
    def seed_everything(seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)




class Config_03_train_lstmfcn:
    project_name = 'BlastoClass_y13-18_3days_288frames_optflow_LK'
    data_path = Config_02_temporalData.output_csvNormalized_file_path
    keyAPIpath = "C:/Users/loren/Documents/keyAPIwandb.txt"
    
    model_name = 'LSTMFCN'
    dataset = "Blasto"
    train_size = 0.8
    val_size = 0.2

    img_size = utils.img_size
    num_classes = utils.num_classes

    # Parametri LSTM
    num_epochs = 2000
    batch_size = 32                  # numero di sequenze prese
    dropout = 0.8
    kernel_sizes = (8,5,3)
    filter_sizes = (128,256,128)
    lstm_size = 8                      # Numero di layer LSTM
    attention = False
    verbose = 2

    # Seed
    seed = utils.seed
    def seed_everything(seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    exp_name = dataset + "," + model_name + "," + str(num_epochs) + "," + str(batch_size)





