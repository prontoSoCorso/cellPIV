''' Configuration file for the project'''

import os
import torch
import random
import numpy as np
import optuna

# Rileva il percorso della cartella "cellPIV" in modo dinamico
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while os.path.basename(parent_dir) != "cellPIV":
    parent_dir = os.path.dirname(parent_dir)

# 0 newPC, 1 lorenzo, 2 AWS
sourceForPath = 0

class user_paths:
    if sourceForPath == 0:
        #Per computer fisso nuovo
        path_excels = parent_dir 
        path_BlastoData = "/home/phd2/Scrivania/CorsoData/blastocisti/"
        #path_BlastoData = "/home/phd2/Scrivania/Data/BlastoData/"      #Per usare solo 2013 e 2014
        #path_BlastoData = "/home/phd2/Scrivania/Data/BlastoDataProva/"  #Solo 20 video
    
    elif sourceForPath == 1:
        #Per computer portatile lorenzo
        path_excels = parent_dir
        path_BlastoData = "C:/Users/loren/Documents/Data/BlastoData/"

    elif sourceForPath == 2:
        #Per AWS
        path_excels = "/home/ec2-user/cellPIV/"
        path_BlastoData = "/mnt/s3bucket/blastocisti/"
    

class utils:
    # Dim
    img_size                    = 500
    num_frames_3Days            = 288
    num_frames_7Days            = 672
    num_classes                 = 2
    project_name                = "BlastoClass_7days_672frames_optflow_LK"

    # Seed everything
    seed = 2024


class Config_00_preprocessing:
    path_old_excel          = os.path.join(user_paths.path_excels, "BlastoLabels.xlsx")
    path_single_csv         = os.path.join(user_paths.path_excels, "_00_preprocessing", "BlastoLabels_singleFile.csv")
    path_singleWithID_csv   = os.path.join(user_paths.path_excels, "_00_preprocessing", "BlastoLabels_singleFileWithID.csv")
    path_double_dish_excel  = os.path.join(user_paths.path_excels, "pz con doppia dish.xlsx")


class Config_01_OpticalFlow:
    #method_optical_flow = "LucasKanade"
    method_optical_flow = "Farneback"

    if method_optical_flow == "LucasKanade":
        # LUCAS KANADE
        # LK parameters
        winSize         = 10
        maxLevelPyramid = 3
        maxCorners      = 300
        qualityLevel    = 0.3
        minDistance     = 10
        blockSize       = 7

        # Var
        img_size                    = utils.img_size
        save_images                 = 0
        num_minimum_frames          = 300
        num_initial_frames_to_cut   = 5
        num_forward_frame           = 4   # Numero di frame per sum_mean_mag
        
    elif method_optical_flow == "Farneback":
        # FARNEBACK
        # Farneback parameters
        pyr_scale = 0.5
        levels = 3
        winSize = 10
        iterations = 4
        poly_n = 5
        poly_sigma = 1.2
        flags = 0

        # Var
        img_size = utils.img_size
        save_images = 0
        num_minimum_frames = 580 #300 per 3 giorni, 390 per 4 giorni, 490 per 5 giorni, 580 per 6 giorni, 680 per 7 giorni
        num_initial_frames_to_cut = 5
        num_forward_frame = 4   # Numero di frame per sum_mean_mag

    else:
        raise SystemExit("\n===== Scegliere un metodo di flusso ottico valido nel config =====\n")



class Config_02_temporalData:
    dict                        = "sum_mean_mag"
    OptFlow                     = Config_01_OpticalFlow.method_optical_flow
    dictAndOptFlowType          = dict + "_" + OptFlow + ".csv"
    temporal_csv_path           = os.path.join(parent_dir, '_02_temporalData', 'final_series', dictAndOptFlowType)
    csv_file_Danilo_path        = Config_00_preprocessing.path_singleWithID_csv
    output_final_csv_path       = os.path.join(user_paths.path_excels, "_02_temporalData", "FinalBlastoLabels.csv")



class Config_02b_normalization:
    # Data
    temporalDataType            = Config_02_temporalData.dict

    #Paths
    csv_file_path                   = Config_02_temporalData.output_final_csv_path
    output_normalized_train_path    = os.path.join(user_paths.path_excels, f"Normalized_train_{temporalDataType}.csv")
    output_normalized_val_path      = os.path.join(user_paths.path_excels, f"Normalized_val_{temporalDataType}.csv")
    output_normalized_test_path     = os.path.join(user_paths.path_excels, f"Normalized_test_{temporalDataType}.csv")

    # Vars
    n_last_colums_check_max = 8

    # Seed
    seed = utils.seed
    def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)













class Config_03_LSTM:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multi_gpu = torch.cuda.device_count() > 1  # Variabile per controllare l'uso di più GPU
    project_name = utils.project_name
    data_path    = Config_02b_normalization.output_csvNormalized_file_path
    test_path    = Config_02b_normalization.test_data_path

    test_dir     = "_04_test"
    
    num_classes = utils.num_classes
    train_size  = 0.8
    val_size    = 0.2

    # Seed
    seed = utils.seed
    def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    # Parametri LSTM
    num_epochs      = 100
    batch_size      = 16
    learning_rate   = 1e-4
    hidden_size     = 128
    num_layers      = 3
    dropout         = 0.25
    bidirectional   = False



class Config_03_LSTM_WithOptuna:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multi_gpu = torch.cuda.device_count() > 1  # Variabile per controllare l'uso di più GPU
    project_name = utils.project_name
    data_path    = Config_02b_normalization.output_csvNormalized_file_path
    test_path    = Config_02b_normalization.test_data_path

    test_dir     = "_04_test"
    
    num_classes = utils.num_classes
    train_size  = 0.8
    val_size    = 0.2

    # Seed
    seed = utils.seed
    def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    # Parametri LSTM
    num_epochs          = [200,300,400]
    batch_size          = [8,16,32,64]  # Numero di sequenze prese
    dropout             = np.arange(0.1, 0.4, 0.05)
    hidden_size         = [32, 64, 128]
    num_layers          = [2,3,4]
    learning_rate       = [random.uniform(1e-4, 1e-3) for _ in range(10)]

    sampler             = optuna.samplers.TPESampler(seed=seed)
    n_startup_trials    = 20
    pruner              = optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials)
    

class Config_03_train_rocket:
    project_name    = utils.project_name
    data_path       = Config_02b_normalization.output_csvNormalized_file_path
    test_path       = Config_02b_normalization.test_data_path

    test_dir     = "_04_test"

    kernels     = [100,300,500,1000,2500,5000,10000] #provato con [50,100,200,300,500,1000,5000,10000,20000]
    val_size   = 0.25
    img_size    = utils.img_size
    num_classes = utils.num_classes

    # Seed
    seed = utils.seed
    def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


