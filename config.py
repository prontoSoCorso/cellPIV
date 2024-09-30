''' Configuration file for the project'''

import torch
import random
import numpy as np
import torch
import optuna

giovanna = False

class user_paths:
    #Per computer fisso
    if giovanna:
        path_BlastoData = "/home/giovanna/Documents/Data/BlastoData/"
    
    else:
    #Per computer portatile
        path_BlastoData = "C:/Users/loren/Documents/Data/BlastoData/"


class utils:
    # Dim
    img_size                    = 500
    num_frames                  = 288
    num_classes                 = 2
    project_name                = "BlastoClass_y13-14_3days_288frames_optflow_LK"

    # Seed everything
    seed = 2024

    # Test dir
    test_dir = "_04_test"


class Config_00_preprocessing:
    path_old_excel          = user_paths.path_BlastoData + "BlastoLabels.xlsx"
    path_single_csv         = user_paths.path_BlastoData + "BlastoLabels_singleFile.csv"
    path_singleWithID_csv   = user_paths.path_BlastoData + "BlastoLabels_singleFileWithID.csv"
    path_double_dish_excel  = user_paths.path_BlastoData + "pz con doppia dish.xlsx"



class Config_01_OpticalFlow:
    # Paths
    project_name        = 'BlastoClass_y13-18_3days_288frames_optflow_LK'
    method_optical_flow = "LucasKanade"

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
    


class Config_02_temporalData:
    #Paths
    csv_file_path                   = Config_00_preprocessing.path_singleWithID_csv
    output_csv_file_path            = user_paths.path_BlastoData + "FinalBlastoLabels.csv"
    output_csvNormalized_file_path  = user_paths.path_BlastoData + "Normalized_Final_BlastoLabels.csv"

    # Data
    temporalDataType = "sum_mean_mag_dict"

    # Vars
    n_last_colums_check_max = 8



class Config_03_LSTM:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_name = utils.project_name
    data_path    = Config_02_temporalData.output_csvNormalized_file_path
    test_dir     = utils.test_dir
    
    num_classes = utils.num_classes
    train_size  = 0.8
    val_size    = 0.2

    # Seed
    seed = utils.seed
    def seed_everything(seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    # Parametri LSTM
    num_epochs      = 500
    batch_size      = 32
    learning_rate   = 1e-3
    hidden_size     = 128
    num_layers      = 4
    dropout         = 0.25
    bidirectional   = False



class Config_03_LSTM_WithOptuna:
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_name = utils.project_name
    data_path    = Config_02_temporalData.output_csvNormalized_file_path
    test_dir     = utils.test_dir
    
    num_classes = utils.num_classes
    train_size  = 0.8
    val_size    = 0.2

    # Seed
    seed = utils.seed
    def seed_everything(seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    # Parametri LSTM
    num_epochs          = [300,400,500]
    batch_size          = [8,16,32,64]  # Numero di sequenze prese
    dropout             = np.arange(0.1, 0.4, 0.05)
    hidden_size         = [32, 64, 128]
    num_layers          = [2,3,4]

    sampler             = optuna.samplers.TPESampler(seed=seed)
    n_startup_trials    = 10
    pruner              = optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials)
    



class Config_03_train_rocket:
    project_name    = utils.project_name
    data_path       = Config_02_temporalData.output_csvNormalized_file_path
    test_dir        = utils.test_dir

    model_name  = 'Rocket'
    dataset     = "Blasto"
    kernels     = [50, 100, 200, 300, 10000]
    test_size   = 0.2
    img_size    = utils.img_size
    num_classes = utils.num_classes

    # Nome dell'esperimento
    exp_name = dataset + "," + model_name

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
    project_name = utils.project_name
    data_path    = Config_02_temporalData.output_csvNormalized_file_path
    test_dir     = utils.test_dir
    
    model_name  = 'LSTMFCN'
    dataset     = "Blasto"
    train_size  = 0.8
    val_size    = 0.2

    img_size    = utils.img_size
    num_classes = utils.num_classes

    # Seed
    seed = utils.seed
    def seed_everything(seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    # Parametri LSTMFCN
    num_epochs      = 250
    batch_size      = 16                  # numero di sequenze prese
    dropout         = 0.2
    kernel_sizes    = (8,5,3) #def: 8,5,3
    filter_sizes    = (256,128,128)
    lstm_size       = 6                      # Numero di layer LSTM
    attention       = False
    verbose         = 2


    # Nome dell'esperimento
    exp_name = dataset + "," + model_name + "," + str(num_epochs) + "," + str(batch_size) + "," + str(kernel_sizes) + "," + str(filter_sizes) + "," + str(lstm_size) + "," + str(attention)


class Config_03_train_lstmfcn_with_optuna:
    data_path   = Config_02_temporalData.output_csvNormalized_file_path
    test_dir    = utils.test_dir

    train_size  = 0.8
    val_size    = 0.2

    img_size    = utils.img_size
    num_classes = utils.num_classes

    # Seed
    seed = utils.seed
    def seed_everything(seed=0):
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
    num_epochs          = 1500

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
    
    in_num_epochs      = 50
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
    test_dir     = utils.test_dir
    
    model_name = 'HIVECOTEV2'
    dataset = "Blasto"
    test_size = 0.1
    img_size = utils.img_size
    num_classes = utils.num_classes

    # Nome dell'esperimento
    exp_name = dataset + "," + model_name

    # Seed
    seed = utils.seed
    def seed_everything(seed=0):
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
    # Input & Output
    project_name = utils.project_name
    data_path = Config_02_temporalData.output_csvNormalized_file_path
    output_dir = user_paths.path_BlastoData
    Norm = False        # Data Normalization
    val_ratio = 0.2     # Propotion of train-set to be used as validation
    print_interval = 10 # Print batch info every this many batches
    test_dir = utils.test_dir
    tensorboard_dir = "_98_ConvTranSummary"

    # Transformers Parameters
    Net_Type = 'C-T'    # choices={'T', 'C-T'}, help="Network Architecture. Convolution (C)", "Transformers (T)") (def = C-T)
    emb_size = 16       # Internal dimension of transformer embeddings (def = 16)
    dim_ff = 256        # Dimension of dense feedforward part of transformer layer (def = 256)
    num_heads = 8       # Number of multi-headed attention heads (def = 8)
    Fix_pos_encode = 'tAPE' # choices={'tAPE', 'Learn', 'None'}, help='Fix Position Embedding'
    Rel_pos_encode = 'eRPE' # choices={'eRPE', 'Vector', 'None'}, help='Relative Position Embedding'

    # Training Parameters/Hyper-Parameters
    epochs = 50        # Number of training epochs
    batch_size = 16     # Training batch size
    lr = 1e-3           # Learning rate
    dropout = 0.2       # Dropout regularization ratio
    val_interval = 2    # Evaluate on validation every XX epochs
    key_metric = 'accuracy' # choices={'loss', 'accuracy', 'precision'}, help='Metric used for defining best epoch'
    num_classes = utils.num_classes
    
    # System
    gpu = 0             # GPU index, -1 for CPU
    console = False     # Optimize printout for console output; otherwise for file
    
    # Seed
    seed = utils.seed
    def seed_everything(seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

