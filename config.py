''' Configuration file for the project'''

import os
import torch
import random
import numpy as np
import cv2

# Rileva il percorso della cartella "cellPIV" in modo dinamico
current_file_path = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(current_file_path)
while os.path.basename(PROJECT_ROOT) != "cellPIV":
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

# 0 newPC, 1 lorenzo, 2 AWS
sourceForPath = 0

print_source = 0
if print_source:
    if sourceForPath==0: 
        to_print = "Workstation Uni"
    elif sourceForPath==1:
        to_print = "PC Lorenzo"
    elif sourceForPath==2:
        to_print = "AWS"
    #print(f"Using path for this computer: {to_print}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class user_paths:
    if sourceForPath == 0:
        #Per computer fisso nuovo
        dataset = os.path.join(PROJECT_ROOT, "datasets")
        path_original_excel = os.path.join(dataset, "DB morpheus UniPV.xlsx")
        path_BlastoData = "/home/phd2/Scrivania/CorsoData/blastocisti"
    
    elif sourceForPath == 1:
        #Per computer portatile lorenzo
        dataset = os.path.join(PROJECT_ROOT, "datasets")
        path_original_excel = os.path.join(dataset, "DB morpheus UniPV.xlsx")
        path_BlastoData = "C:/Users/loren/Documents/Data/BlastoData"

    elif sourceForPath == 2:
        #Per AWS
        dataset = "/home/ec2-user/cellPIV/"
        path_BlastoData = "/mnt/s3bucket/blastocisti"


class utils:
    # Dim
    img_size                    = 500
    framePerHour                = 4
    framePerDay                 = framePerHour*24

    def num_frames_by_days(num_days):
        tot_frames = utils.framePerDay*num_days+1
        return tot_frames

    num_classes                 = 2
    project_name                = "BlastoClass_3days_optflow_Farneback"
    hours2cut                   = 0
    start_frame                 = framePerHour*hours2cut

    # Seed everything
    seed = 2025


class Config_00_preprocessing:
    # dataPreparation
    input_dir_pdb_files = "/media/phd2/My Passport/ScopeData"
    output_dir_extracted_pdb_files = "/home/phd2/Scrivania/CorsoData/ScopeData_extracted"
    log_file_pdb_extraction = "/home/phd2/Scrivania/CorsoData/estrazione_log.txt"

    src_dir_extracted_pdb = output_dir_extracted_pdb_files
    dest_dir_extracted_equator = "/home/phd2/Scrivania/CorsoData/ScopeData_equator"

    src_dir_extracted_equator = dest_dir_extracted_equator
    dest_dir_time_conversion = "/home/phd2/Scrivania/CorsoData/ScopeData_time_conversion"
    
    # Percorso della directory con i video equatoriali
    path_main_folder = dest_dir_extracted_equator

    # Path cartella in cui salvare tutti i "valid_wells" selezionati dalle immagini, principalemente dai timing di acquisizione
    valid_wells_file = os.path.join(user_paths.dataset, 'valid_wells_acquisition_times.csv')

    # Preprocessing
    path_original_excel     = user_paths.path_original_excel
    #path_original_excel     = os.path.join(user_paths.path_excels, "BlastoLabels.xlsx")
    path_addedID_csv        = os.path.join(user_paths.dataset, "DB_Morpheus_withID.csv")
    path_double_dish_excel  = os.path.join(user_paths.dataset, "pz con doppia dish.xlsx")
    filtered_blasto_dataset = os.path.join(user_paths.dataset, "filtered_blasto_dataset.csv")

    # Percorso della directory di destinazione
    dest_dir_blastoData = user_paths.path_BlastoData

class Config_01_OpticalFlow:
    method_optical_flow     = "Farneback"   # "LucasKanade/Farneback"
    #method_optical_flow    = "LucasKanade"   # "LucasKanade/Farneback"
    pickle_dir              = os.path.join(user_paths.dataset, method_optical_flow, 'pickles')

    # Settings
    save_metrics = False
    save_overlay_optical_flow = False
    save_final_data = True

    # Var
    img_size                    = utils.img_size
    num_minimum_frames          = utils.framePerDay*3   # Numero minimo di frame per considerare il video (3 giorni)
    num_initial_frames_to_cut   = utils.start_frame     # Numero di frame iniziali da tagliare (es. 0h, 1h, 2h, 3h)
    num_forward_frame           = 4                     # Numero di frame per sum_mean_mag

    # LUCAS KANADE
    # LK parameters
    winSize_LK      = 13
    maxLevelPyramid = 4
    maxCorners      = 300
    qualityLevel    = 0.05
    minDistance     = 5
    blockSize       = 7

    # FARNEBACK
    # Farneback parameters

    # General analysis
    """
    FINAL PARAMS: 
    pyr_scale        = 0.5
    levels           = 4
    winSize_Farneback= 25                               # più grande
    iterations       = 3                                # più iterazioni
    poly_n           = 5                                # vicinato maggiore
    poly_sigma       = 1.2                              # filtro gaussiano più largo
    flags            = 0
    """
    pyr_scale = 0.5
    levels = 4
    winSize_Farneback = 25
    iterations = 3
    poly_n = 5
    poly_sigma = 1.2
    flags = 0

    # Singolo video
    single_video = False
    if single_video:
        pyr_scale        = 0.5
        levels           = 4
        winSize_Farneback= 25                               # più grande
        iterations       = 3                                # più iterazioni
        poly_n           = 5                                # vicinato maggiore
        poly_sigma       = 1.2                              # filtro gaussiano più largo
        flags            = cv2.OPTFLOW_FARNEBACK_GAUSSIAN

    base_out_example = f"/home/phd2/Scrivania/CorsoData/opticalFlowExamples{method_optical_flow}" 
    if method_optical_flow == "Farneback":
        output_path_optical_flow_images = f"{base_out_example}_{str(winSize_Farneback)}_{str(levels)}_{str(pyr_scale)}_{str(iterations)}_{str(poly_n)}_{str(poly_sigma)}"
    elif method_optical_flow == "LucasKanade":
        output_path_optical_flow_images = f"{base_out_example}_{str(winSize_LK)}_{str(maxLevelPyramid)}_{str(maxCorners)}"
    else:
        raise SystemExit("\n===== Scegliere un metodo di flusso ottico valido nel config =====\n")


class Config_02_processTemporalData:
    method_optical_flow     = "Farneback"   # "LucasKanade/Farneback"
    pickle_dir              = os.path.join(user_paths.dataset, method_optical_flow, 'pickles')
    dict                    = "vorticity"  # mean_magnitude / sum_mean_mag / vorticity / hybrid

    # File pickle e CSV da caricare per il processamento dei dati raw del flusso ottico
    pickle_files            = [f"{dict}.pkl"]   # es. ["sum_mean_mag.pkl"] oppure [] per tutti
    reference_file_path     = Config_00_preprocessing.filtered_blasto_dataset  # File che ho ottenuto dal preprocessing degli excel (singolo csv con ID)
    acquisition_times_path  = Config_00_preprocessing.valid_wells_file  # File con i tempi di acquisizione
    
    # Path del csv finale che contiene gli identificativi dei video, la classe, i metadati e tutti i valori delle serie temporali allineate
    out_pickle_dir          = os.path.join(user_paths.dataset, method_optical_flow, "pickles_preprocessed")
    final_csv_path          =  os.path.join(user_paths.dataset, method_optical_flow, "FinalDataset.csv")


class Config_02c_splitAndNormalization:
    method_optical_flow = "Farneback"   #LucasKanade / Farneback

    # Data
    temporalDataType = Config_02_processTemporalData.dict
    train_size = 0.7
    embedding_type=""   # "umap" OR "tsne"
    save_normalization_example_single_pt=False
    per_series_normalization = True
    mean_data_visualization=False
    specific_patient_to_analyse=42
    mean_data_visualization_stratified=False
    path_original_excel = user_paths.path_original_excel

    # Per gestire dati a N giorni a partire dalla i-esima ora con limiti normalizzazione
    days_to_consider = [1]        # Imposta il numero di giorni da considerare (1, 3, 5, o 7)
    inf_quantile = 0.01
    sup_quantile = 0.99
    initial_hours_to_cut = 0    # Remember that I already cut the first hour
    initial_frames_to_cut = initial_hours_to_cut*utils.framePerHour
    start_time = initial_frames_to_cut+utils.start_frame

    # Paths file completo
    csv_file_path = os.path.join(user_paths.dataset, method_optical_flow, "FinalDataset.csv")

    # output dir for umap/tsne
    visual_output_dir_base = os.path.join(PROJECT_ROOT, "_02c_splitAndnormalization", method_optical_flow, "dim_reduction_files")
    
    # Base path generico per i file normalizzati
    @staticmethod
    def get_normalized_base_path(days_to_consider, method_optical_flow=method_optical_flow, small_subsets=False):
        if small_subsets:
            subsets_base_path = os.path.join(user_paths.dataset, method_optical_flow, "small_subsets")
        else:
            subsets_base_path = os.path.join(user_paths.dataset, method_optical_flow, "subsets")
        return os.path.join(subsets_base_path, 
                            f"Normalized_{Config_02c_splitAndNormalization.temporalDataType}_{days_to_consider}Days")

    # Metodo per ottenere i percorsi in base ai giorni selezionati
    @staticmethod
    def get_paths(days_to_consider, small_subsets=False):
        """
        Ottiene i percorsi di train, validation e test in base al numero di giorni selezionati.

        :param days_to_consider: Numero di giorni da considerare (1, 3, 5, o 7).
        :return: Tuple con i percorsi di train, validation e test.
        """
        base_path = Config_02c_splitAndNormalization.get_normalized_base_path(days_to_consider=days_to_consider, 
                                                                      method_optical_flow=Config_02c_splitAndNormalization.method_optical_flow,
                                                                      small_subsets=small_subsets)
        train_path = f"{base_path}_train.csv"
        val_path = f"{base_path}_val.csv"
        test_path = f"{base_path}_test.csv"
        return train_path, val_path, test_path

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
    method_optical_flow = "Farneback"
    days_label = 3
    project_name = utils.project_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multi_gpu = torch.cuda.device_count() > 1  # Variabile per controllare l'uso di più GPU
    num_classes = utils.num_classes
    img_size = utils.img_size
    seed = utils.seed
    num_labels = 2
    Data_shape = (1,97) #variabile di base, verrà aggiornata in ConvTran
    output_model_base_dir = os.path.join(PROJECT_ROOT, "_04_test", "best_models", method_optical_flow)
    save_plots = True
    output_dir_plots = os.path.join(PROJECT_ROOT, "_03_train", "test_results_after_training", method_optical_flow)

    @staticmethod
    def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    # Metodo per ottenere i percorsi in base ai giorni selezionati
    @staticmethod
    def get_paths(days_to_consider, small_subsets=False):
        """
        Ottiene i percorsi di train, validation e test in base al numero di giorni selezionati.

        :param days_to_consider: Numero di giorni da considerare (1, 3, 5, o 7).
        :return: Tuple con i percorsi di train, validation e test.
        """
        base_path = Config_02c_splitAndNormalization.get_normalized_base_path(days_to_consider=days_to_consider, 
                                                                      method_optical_flow=Config_03_train.method_optical_flow,
                                                                      small_subsets=small_subsets)
        train_path = f"{base_path}_train.csv"
        val_path = f"{base_path}_val.csv"
        test_path = f"{base_path}_test.csv"
        return train_path, val_path, test_path
    

    # ROCKET
    kernel_number_ROCKET     = [5000, 10000, 15000, 20000] #provato con [50,100,200,300,500,1000,5000,10000,20000]
    type_model_classification = "RF"    #or "LR" or "XGB"
    most_important_metric = "balanced_accuracy"
    
    # LSTM-FCN
    num_epochs_FCN      = 300
    batch_size_FCN      = 16
    dropout_FCN         = 0.3
    kernel_sizes_FCN    = "7,5,3"           # def: 7,5,3
    filter_sizes_FCN    = "128,256,128"
    lstm_size_FCN       = 4
    attention_FCN       = False
    verbose_FCN         = 2
    learning_rate_FCN   = 1e-4
    hidden_size_FCN     = 128
    bidirectionale_FCN  = False
    num_layers_FCN      = 4
    final_epochs_FCN    = 500


    # ConvTran
    # ConvTran - Input & Output                                  
    output_dir_convtran = user_paths.dataset
    Norm                = False        # Data Normalization
    print_interval      = 50 # Print batch info every this many batches
    # ConvTran - Transformers Parameters
    Net_Type            = 'C-T'    # choices={'T', 'C-T'}, help="Network Architecture. Convolution (C)", "Transformers (T)") (def = C-T)
    emb_size_convtran   = 128       # Internal dimension of transformer embeddings (def = 16)
    kernel_len_emb      = 9         # Kernel size of initial Conv1D embedding layer (def = 9)
    dim_ff              = emb_size_convtran*2       # Dimension of dense feedforward part of transformer layer (def = 256)
    num_heads_convtran  = 8       # Number of multi-headed attention heads (def = 8)
    Fix_pos_encode      = 'tAPE' # choices={'tAPE', 'Learn', 'None'}, help='Fix Position Embedding'
    Rel_pos_encode      = 'eRPE' # choices={'eRPE', 'Vector', 'None'}, help='Relative Position Embedding'
    # ConvTran - Training Parameters/Hyper-Parameters
    epochs_convtran     = 150        # Number of training epochs
    batch_size_convtran = 32     # Training batch size
    lr_convtran         = 5e-4           # Learning rate
    dropout_convtran    = 0.3       # Dropout regularization ratio
    num_classes         = utils.num_classes
    num_labels          = num_classes
    # ConvTran - scheduler and Early Stopping
    patience_convtran   = 50    # Number of epochs with no improvement after which training will be stopped
    scheduler_patience_convtran   = 10    # Number of epochs with no improvement after which learning rate will be reduced
    scheduler_factor_convtran     = 0.75    # Factor by which the learning rate will be reduced
    # ConvTran - System
    gpu                 = -1             # GPU index, -1 for CPU
    console             = True     # Optimize printout for console output; otherwise for file




class Config_03_train_with_optimization(Config_03_train):
    # Enable/disable test evaluation
    run_test_evaluation = True
    
    # Optuna optimization control
    optimize_with_optuna = False
    
    # --- ROCKET search space ---
    rocket_kernels_options = [5000, 7500, 10000, 12500]
    rocket_classifier_options = ["RF", "xgb"]
    optuna_n_trials_ROCKET = 20
    
    # --- LSTM-FCN search space ---
    lstm_size_options = [4, 8, 16, 32]
    filter_sizes_options = ["64,128,64", "128,256,128", "256,512,256"]
    kernel_sizes_options = ["5,3,2", "8,5,3", "10,7,5"]
    dropout_range = (0.1, 0.4)
    num_layers_range = (1, 6)
    batch_size_options = [16, 32, 64]
    learning_rate_range = (1e-5, 1e-3)
    early_stopping_patience = 60
    optuna_num_epochs = 150
    optuna_n_trials_LSTMFCN = 300

    # --- ConvTran search space ---
    convtran_emb_size_options   = [64, 128, 256]
    convtran_dim_ff_options     = [32, 64, 128, 256]
    convtran_num_heads_options  = [4, 8, 16]
    convtran_dropout_range      = (0.1, 0.5)
    convtran_learning_rate_range= (1e-5, 1e-3)
    convtran_batch_size_options = [16, 32, 64]
    convtran_patience_options   = 50
    convtran_epochs_options     = 150
    optuna_n_trials_ConvTran    = 300



class Config_04_test:
    method_optical_flow = "Farneback"
    base_model_path = Config_03_train.output_model_base_dir  #os.path.join(PROJECT_ROOT, "_04_test", "best_models", method_optical_flow)
    base_output_stratified_path = os.path.join(PROJECT_ROOT, "_04_test", "stratified_test_results", method_optical_flow)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = utils.num_classes
    img_size = utils.img_size
    seed = utils.seed
    num_labels = 2

    @staticmethod
    def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    # Metodo per ottenere i percorsi in base ai giorni selezionati
    @staticmethod
    def get_paths(days_to_consider):
        """
        Ottiene i percorsi di train, validation e test in base al numero di giorni selezionati.

        :param days_to_consider: Numero di giorni da considerare (1, 3, 5, o 7).
        :return: Tuple con i percorsi di train, validation e test.
        """
        base_path = Config_02c_splitAndNormalization.get_normalized_base_path(days_to_consider=days_to_consider, 
                                                                      method_optical_flow=Config_04_test.method_optical_flow)
        train_path = f"{base_path}_train.csv"
        val_path = f"{base_path}_val.csv"
        test_path = f"{base_path}_test.csv"
        return train_path, val_path, test_path