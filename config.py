''' Configuration file for the project'''

import os
import torch
import random
import numpy as np

# Rileva il percorso della cartella "cellPIV" in modo dinamico
current_file_path = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(current_file_path)
while os.path.basename(PROJECT_ROOT) != "cellPIV":
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

# 0 newPC, 1 lorenzo, 2 AWS
sourceForPath = 0
print(sourceForPath)

class user_paths:
    if sourceForPath == 0:
        #Per computer fisso nuovo
        path_excels = PROJECT_ROOT 
        path_BlastoData = "/home/phd2/Scrivania/CorsoData/blastocisti/"
        #path_BlastoData = "/home/phd2/Scrivania/Data/BlastoData/"      #Per usare solo 2013 e 2014
        #path_BlastoData = "/home/phd2/Scrivania/Data/BlastoDataProva/"  #Solo 20 video
    
    elif sourceForPath == 1:
        #Per computer portatile lorenzo
        path_excels = PROJECT_ROOT
        path_BlastoData = "C:/Users/loren/Documents/Data/BlastoData/"

    elif sourceForPath == 2:
        #Per AWS
        path_excels = "/home/ec2-user/cellPIV/"
        path_BlastoData = "/mnt/s3bucket/blastocisti/"
    

class utils:
    # Dim
    img_size                    = 500
    framePerDay                 = 96
    num_frames_3Days            = framePerDay * 3
    num_frames_5Days            = framePerDay * 5
    num_frames_7Days            = framePerDay * 7
    num_classes                 = 2
    project_name                = "BlastoClass_7days_672frames_optflow_LK"

    # Seed everything
    seed = 2024


class Config_00_preprocessing:
    path_original_excel     = os.path.join(user_paths.path_excels, "DB morpheus UniPV.xlsx")
    #path_original_excel     = os.path.join(user_paths.path_excels, "BlastoLabels.xlsx")
    path_addedID_csv        = os.path.join(user_paths.path_excels, "_00b_preprocessing_excels", "DB_Morpheus_withID.csv")
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
        num_minimum_frames = 300
        num_initial_frames_to_cut = 5
        num_forward_frame = 4   # Numero di frame per sum_mean_mag

    else:
        raise SystemExit("\n===== Scegliere un metodo di flusso ottico valido nel config =====\n")



class Config_02_temporalData:
    dict                        = "sum_mean_mag"
    OptFlow                     = Config_01_OpticalFlow.method_optical_flow
    type_files                  = "files_7Days_Farneback"
    dict_in                     = dict + "_dict_" + OptFlow + ".pkl"
    
    path_pkl                    = os.path.join(type_files, dict_in)
    dictAndOptFlowType          = dict + "_" + OptFlow + ".csv"

    # Path in cui salvo il file csv che ottengo leggendo i pkl delle serie temporali e che sarà poi quello usato per creare il csv finale
    temporal_csv_path           = os.path.join(PROJECT_ROOT, '_02_temporalData', 'final_series', dictAndOptFlowType)
    csv_file_Danilo_path        = Config_00_preprocessing.path_addedID_csv  # File che ho ottenuto dal preprocessing degli excel (singolo csv con ID)
    # Path del csv finale che contiene gli identificativi dei video, la classe e tutti i valori delle serie temporali
    final_csv_path              = os.path.join(user_paths.path_excels, "_02_temporalData", "FinalBlastoLabels.csv")


class Config_02b_normalization:
    # Data
    temporalDataType            = Config_02_temporalData.dict

    # Per gestire dati a 3, 5 o 7 giorni
    Only3Days = 0
    Only5Days = 1

    #Paths 3 Days
    csv_file_path               = Config_02_temporalData.final_csv_path
    normalized_train_path_3Days = os.path.join(user_paths.path_excels, f"Normalized_train_3Days_{temporalDataType}.csv")
    normalized_val_path_3Days   = os.path.join(user_paths.path_excels, f"Normalized_val_3Days_{temporalDataType}.csv")
    normalized_test_path_3Days  = os.path.join(user_paths.path_excels, f"Normalized_test_3Days_{temporalDataType}.csv")

    #Paths 5 Days
    csv_file_path               = Config_02_temporalData.final_csv_path
    normalized_train_path_5Days = os.path.join(user_paths.path_excels, f"Normalized_train_5Days_{temporalDataType}.csv")
    normalized_val_path_5Days   = os.path.join(user_paths.path_excels, f"Normalized_val_5Days_{temporalDataType}.csv")
    normalized_test_path_5Days  = os.path.join(user_paths.path_excels, f"Normalized_test_5Days_{temporalDataType}.csv")

    #Paths 7 Days
    csv_file_path               = Config_02_temporalData.final_csv_path
    normalized_train_path_7Days = os.path.join(user_paths.path_excels, f"Normalized_train_7Days_{temporalDataType}.csv")
    normalized_val_path_7Days   = os.path.join(user_paths.path_excels, f"Normalized_val_7Days_{temporalDataType}.csv")
    normalized_test_path_7Days  = os.path.join(user_paths.path_excels, f"Normalized_test_7Days_{temporalDataType}.csv")

    # Seed
    seed = utils.seed
    def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)



class Config_03_train:
    project_name    = utils.project_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multi_gpu = torch.cuda.device_count() > 1  # Variabile per controllare l'uso di più GPU
    num_classes = utils.num_classes
    img_size    = utils.img_size
    seed = utils.seed
    test_dir    = "_04_test"
    def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    # Dizionari per gestire percorsi in modo dinamico
    frame_options = {
        "3Days": {
            "train_path": Config_02b_normalization.normalized_train_path_3Days,
            "val_path"  : Config_02b_normalization.normalized_val_path_3Days,
            "test_path" : Config_02b_normalization.normalized_test_path_3Days,
        },
        "5Days": {
            "train_path": Config_02b_normalization.normalized_train_path_5Days,
            "val_path"  : Config_02b_normalization.normalized_val_path_5Days,
            "test_path" : Config_02b_normalization.normalized_test_path_5Days,
        },
        "7Days": {
            "train_path": Config_02b_normalization.normalized_train_path_7Days,
            "val_path"  : Config_02b_normalization.normalized_val_path_7Days,
            "test_path" : Config_02b_normalization.normalized_test_path_7Days,
        }
    }

    # Metodo per ottenere i percorsi in base ai giorni selezionati
    @classmethod
    def get_paths(cls, selected_days="7Days"):
        if selected_days in cls.frame_options:
            paths = cls.frame_options[selected_days]
            return paths["train_path"], paths["val_path"], paths["test_path"]
        else:
            raise ValueError(f"Opzione non valida per il numero di giorni: {selected_days}")
    


    # ROCKET
    kernels     = [100,300,500,1000,2500,5000,7500,10000,12500,15000] #provato con [50,100,200,300,500,1000,5000,10000,20000]


    # LSTM-FCN
    num_epochs_FCN      = 1000
    batch_size_FCN      = 16                     # numero di sequenze prese (con 16 arrivo a 84%)
    dropout_FCN         = 0.3
    kernel_sizes_FCN    = (8,5,3) #def: 8,5,3
    filter_sizes_FCN    = (128,256,128)
    lstm_size_FCN       = 4                      # Numero di layer LSTM (con 4 arrivo a 84%)
    attention_FCN       = False
    verbose_FCN         = 2
    learning_rate_FCN   = 1e-4
    hidden_size_FCN     = 128
    bidirectionale_FCN  = False
    num_layers_FCN      = 4
    final_epochs_FCN    = 500


    # ConvTran
    # ConvTran - Input & Output                                  
    output_dir      = user_paths.path_excels
    Norm            = False        # Data Normalization
    val_ratio       = 0.2     # Propotion of train-set to be used as validation
    print_interval  = 50 # Print batch info every this many batches
    # ConvTran - Transformers Parameters
    Net_Type        = 'C-T'    # choices={'T', 'C-T'}, help="Network Architecture. Convolution (C)", "Transformers (T)") (def = C-T)
    emb_size        = 128       # Internal dimension of transformer embeddings (def = 16)
    dim_ff          = emb_size*2       # Dimension of dense feedforward part of transformer layer (def = 256)
    num_heads       = 8       # Number of multi-headed attention heads (def = 8)
    Fix_pos_encode  = 'tAPE' # choices={'tAPE', 'Learn', 'None'}, help='Fix Position Embedding'
    Rel_pos_encode  = 'eRPE' # choices={'eRPE', 'Vector', 'None'}, help='Relative Position Embedding'
    # ConvTran - Training Parameters/Hyper-Parameters
    epochs          = 100        # Number of training epochs
    batch_size      = 16     # Training batch size
    lr              = 1e-3           # Learning rate
    dropout         = 0.2       # Dropout regularization ratio
    val_interval    = 1    # Evaluate on validation every XX epochs
    key_metric      = 'accuracy' # choices={'loss', 'accuracy', 'precision'}, help='Metric used for defining best epoch'
    num_classes     = utils.num_classes
    # ConvTran - Add Learning Rate Scheduler
    scheduler_patience  = 5    # Number of epochs with no improvement after which learning rate will be reduced
    scheduler_factor    = 0.5    # Factor by which the learning rate will be reduced
    # ConvTran - System
    gpu             = -1             # GPU index, -1 for CPU
    console         = False     # Optimize printout for console output; otherwise for file


