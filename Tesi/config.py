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

# 0 giovanna, 1 lorenzo, 2 AWS, 3 newPC
sourceForPath = 3

class user_paths:
    if sourceForPath == 0:
        #Per computer fisso giovanna
        path_excels = parent_dir 
        path_BlastoData = "/home/giovanna/Desktop/CorsoData/blastocisti/"  #Tutto il dataset fino al 2018 incluso
        #path_BlastoData = "/home/giovanna/Documents/Data/BlastoData/"      #Per usare solo 2013 e 2014
        #path_BlastoData = "/home/giovanna/Documents/Data/BlastoDataProva/"  #Solo 20 video
    
    elif sourceForPath == 1:
        #Per computer portatile lorenzo
        path_excels = parent_dir
        path_BlastoData = "C:/Users/loren/Documents/Data/BlastoData/"

    elif sourceForPath == 2:
        #Per AWS
        path_excels = "/home/ec2-user/cellPIV/"
        path_BlastoData = "/mnt/s3bucket/blastocisti/"

    elif sourceForPath == 3:
        #Per computer fisso nuovo
        path_excels = parent_dir 
        path_BlastoData = "/home/phd2/Scrivania/CorsoData/blastocisti/"
        #path_BlastoData = "/home/phd2/Scrivania/Data/BlastoData/"      #Per usare solo 2013 e 2014
        #path_BlastoData = "/home/phd2/Scrivania/Data/BlastoDataProva/"  #Solo 20 video
    

class utils:
    # Dim
    img_size                    = 500
    num_frames                  = 288
    num_classes                 = 2
    project_name                = "BlastoClass_y13-14_3days_288frames_optflow_LK"

    # Seed everything
    seed = 2024


class Config_00_preprocessing:
    path_old_excel          = user_paths.path_excels + "BlastoLabels.xlsx"
    path_single_csv         = user_paths.path_excels + "BlastoLabels_singleFile.csv"
    path_singleWithID_csv   = user_paths.path_excels + "BlastoLabels_singleFileWithID.csv"
    path_double_dish_excel  = user_paths.path_excels + "pz con doppia dish.xlsx"


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
    # Data
    temporalDataType = "sum_mean_mag_dict" # [sum_mean_mag_dict, hybrid_dict, vorticity_dict, mean_magnitude_dict]    
    normalization_type = "TrainTest_ALL" # [PerPatient, TrainTest_ALL, None]

    #Paths
    csv_file_path                   = Config_00_preprocessing.path_singleWithID_csv
    output_csv_file_path            = user_paths.path_excels + "FinalBlastoLabels.csv"
    output_csvNormalized_file_path  = user_paths.path_excels + f"Normalized_Final_BlastoLabels_{temporalDataType}.csv"
    
    output_csvNormalized_AllTrain_file_path  = user_paths.path_excels + f"Normalized_AllTrain_Final_BlastoLabels_{temporalDataType}.csv"
    output_csvNormalized_AllVal_file_path  = user_paths.path_excels + f"Normalized_AllVal_Final_BlastoLabels_{temporalDataType}.csv"
    
    train_val_data_path     = output_csvNormalized_file_path            # Se ho fatto normalizzazione per paziente
    train_data_path         = output_csvNormalized_AllTrain_file_path   # File che ottengo se faccio prima lo split e poi la normalizzazione di train e val, file di train
    val_data_path           = output_csvNormalized_AllVal_file_path     # File che ottengo se faccio prima lo split e poi la normalizzazione di train e val, file di val

    output_csv_file_path_test       = user_paths.path_excels + "FinalBlastoLabels_test.csv"
    test_data_path                  = user_paths.path_excels + f"Normalized_Final_BlastoLabels_test_{temporalDataType}.csv"
    test_NormALL_data_path          = user_paths.path_excels + f"Normalized_ALLTest_Final_BlastoLabels_{temporalDataType}.csv"

    # Vars
    n_last_colums_check_max = 8
    do_only_normalization = True

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
    data_path    = Config_02_temporalData.output_csvNormalized_file_path
    test_path    = Config_02_temporalData.test_data_path

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
    data_path    = Config_02_temporalData.output_csvNormalized_file_path
    test_path    = Config_02_temporalData.test_data_path

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
    data_path       = Config_02_temporalData.output_csvNormalized_file_path
    test_path       = Config_02_temporalData.test_data_path

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


class Config_03_train_rocket_normALL:
    data_path_train = Config_02_temporalData.output_csvNormalized_AllTrain_file_path
    data_path_val   = Config_02_temporalData.output_csvNormalized_AllVal_file_path
    test_path       = Config_02_temporalData.test_NormALL_data_path

    test_dir     = "_04_test"

    model_name  = 'Rocket'
    dataset     = "Blasto"
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




class Config_03_train_lstmfcn:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multi_gpu = torch.cuda.device_count() > 1  # Variabile per controllare l'uso di più GPU
    
    data_path    = Config_02_temporalData.output_csvNormalized_file_path
    test_path    = Config_02_temporalData.test_data_path

    test_dir     = "_04_test"
    
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

    # Parametri LSTMFCN
    num_epochs      = 100
    batch_size      = 16                     # numero di sequenze prese
    dropout         = 0.3
    kernel_sizes    = (8,5,3) #def: 8,5,3
    filter_sizes    = (128,256,128)
    lstm_size       = 4                      # Numero di layer LSTM
    attention       = False
    verbose         = 2

    learning_rate   = 1e-4
    hidden_size = 128
    bidirectionale = False
    num_classes = 2
    num_layers = 4


class Config_03_train_lstmfcn_with_optuna:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multi_gpu = torch.cuda.device_count() > 1  # Variabile per controllare l'uso di più GPU
    
    data_path   = Config_02_temporalData.output_csvNormalized_file_path
    test_path   = Config_02_temporalData.test_data_path

    test_dir     = "_04_test"

    train_size  = 0.8
    val_size    = 0.2

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

    # Parametri Optuna
    pruner = optuna.pruners.MedianPruner()
    sampler = optuna.samplers.TPESampler(seed=seed_everything(seed))
    
    # Parametri LSTMFCN
    min_num_epochs      = 50
    max_num_epochs      = 300
    num_epochs          = 300

    batch_size          = [16,32]            # numero di sequenze prese
    
    min_dropout         = 0.1
    max_dropout         = 0.4

    kernel_sizes        = [(8,5,3)]
    filter_sizes        = [(128,256,128), (256,128,128)]

    min_lstm_size       = 4                     # Numero di layer LSTM
    max_lstm_size       = 8                     

    attention           = False
    verbose             = 2

    '''
    Con questa ricerca:
    
    in_num_epochs       = 50
    max_num_epochs      = 300
    num_epochs          = 500

    batch_size          = [16,32,64]            # numero di sequenze prese
    
    min_dropout         = 0.1
    max_dropout         = 0.4

    kernel_sizes        = [(16,8,4), (8,5,3)]
    filter_sizes        = [(128,256,128), (64,128,64), (256,128,128)]

    min_lstm_size       = 4                     # Numero di layer LSTM
    max_lstm_size       = 8                     

    attention           = False
    verbose             = 2
     
    migliori parametri trovati con accuratezza di 0.63636:
    Miglior trial: {'batch_size': 16, 'dropout': 0.2, 'kernel_sizes': (8, 5, 3), 'filter_sizes': (256, 128, 128), 'lstm_size': 6}
    '''


class Config_03_train_hivecote:
    project_name = utils.project_name
    data_path    = Config_02_temporalData.output_csvNormalized_file_path
    test_path    = Config_02_temporalData.test_data_path

    test_dir     = "_04_test"
    
    model_name = 'HIVECOTEV2'
    dataset = "Blasto"
    img_size = utils.img_size
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
    
    
    # Parametri specifici per la configurazione dei sotto-modelli utilizzati da HiveCoteV2
    stc_params = {"n_shapelet_samples": 10000, "max_shapelets": 10}  # Shapelet Transform Classifier parameters
    drcif_params = {"n_estimators": 500}  # DrCIF parameters
    arsenal_params = {"num_kernels": 2000, "n_estimators": 25}  # Arsenal parameters
    tde_params = {"n_parameter_samples": 25, "max_ensemble_size": 10}  # Temporal Dictionary Ensemble parameters
    
    '''
    # Parametri specifici per la configurazione dei sotto-modelli utilizzati da HiveCoteV2
    stc_params = {"n_shapelet_samples": 1, "max_shapelets": 1}  # Shapelet Transform Classifier parameters
    drcif_params = {"n_estimators": 1}  # DrCIF parameters
    arsenal_params = {"num_kernels": 1, "n_estimators": 1}  # Arsenal parameters
    tde_params = {"n_parameter_samples": 1, "max_ensemble_size": 1}  # Temporal Dictionary Ensemble parameters
    '''
    

class Config_03_train_ConvTran:
    #Paths
    temporalDataType         = Config_02_temporalData.dict
    csv_file_path            = Config_02_temporalData.output_final_csv_path
    normalized_train_path    = os.path.join(user_paths.path_excels, f"Normalized_train_{temporalDataType}.csv")
    normalized_val_path      = os.path.join(user_paths.path_excels, f"Normalized_val_{temporalDataType}.csv")
    normalized_test_path     = os.path.join(user_paths.path_excels, f"Normalized_test_{temporalDataType}.csv")

    # Input & Output
    test_dir     = "_04_test"                                               
    output_dir = user_paths.path_excels
    Norm = False        # Data Normalization
    val_ratio = 0.2     # Propotion of train-set to be used as validation
    print_interval = 10 # Print batch info every this many batches
    
    # Transformers Parameters
    Net_Type = 'C-T'    # choices={'T', 'C-T'}, help="Network Architecture. Convolution (C)", "Transformers (T)") (def = C-T)
    emb_size = 64       # Internal dimension of transformer embeddings (def = 16)
    dim_ff = 128        # Dimension of dense feedforward part of transformer layer (def = 256)
    num_heads = 8       # Number of multi-headed attention heads (def = 8)
    Fix_pos_encode = 'tAPE' # choices={'tAPE', 'Learn', 'None'}, help='Fix Position Embedding'
    Rel_pos_encode = 'eRPE' # choices={'eRPE', 'Vector', 'None'}, help='Relative Position Embedding'

    # Training Parameters/Hyper-Parameters
    epochs = 100        # Number of training epochs
    batch_size = 16     # Training batch size
    lr = 1e-3           # Learning rate
    dropout = 0.2       # Dropout regularization ratio
    val_interval = 2    # Evaluate on validation every XX epochs
    key_metric = 'accuracy' # choices={'loss', 'accuracy', 'precision'}, help='Metric used for defining best epoch'
    num_classes = utils.num_classes

    # Add Learning Rate Scheduler
    scheduler_patience = 5    # Number of epochs with no improvement after which learning rate will be reduced
    scheduler_factor = 0.5    # Factor by which the learning rate will be reduced
    
    # System
    gpu = -1             # GPU index, -1 for CPU
    console = False     # Optimize printout for console output; otherwise for file
    
    # Seed
    seed = utils.seed
    def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)



class Config_03_KNN:
    # Seed
    seed = utils.seed
    def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)



class Config_04_test:
    if Config_02_temporalData.normalization_type == "PerPatient":
        test_path = Config_02_temporalData.test_data_path
    else:
        test_path = Config_02_temporalData.test_NormALL_data_path
    kernel = 300

    test_dir = "_04_test"


class paths_for_models:
    data_path_train = Config_02_temporalData.output_csvNormalized_AllTrain_file_path
    data_path_val   = Config_02_temporalData.output_csvNormalized_AllVal_file_path
    test_path       = Config_02_temporalData.test_NormALL_data_path
    test_dir        = "_04_test"