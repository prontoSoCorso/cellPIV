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
        path_BlastoData = "/home/phd2/Scrivania/CorsoData/blastocisti_small_batch"
    
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
        tot_frames = utils.framePerDay*num_days
        return tot_frames

    num_classes                 = 2
    project_name                = "BlastoClass_3days_optflow_Farneback"
    hours2cut                   = 1
    start_frame                 = framePerHour*hours2cut

    # Seed everything
    seed = 2025


class Config_00_preprocessing:
    # dataPreparation
    input_dir_pdb_files = "/home/phd2/Scrivania/CorsoData/ScopeData"
    output_dir_extracted_pdb_files = "/home/phd2/Scrivania/CorsoData/ScopeData_extracted"
    log_file_pdb_extraction = "/home/phd2/Scrivania/CorsoData/estrazione_log.txt"

    src_dir_extracted_pdb = output_dir_extracted_pdb_files
    dest_dir_extracted_equator = "/home/phd2/Scrivania/CorsoData/ScopeData_equator"
    
    # Percorso della directory con i video equatoriali
    path_main_folder = dest_dir_extracted_equator

    # Preprocessing
    path_original_excel     = user_paths.path_original_excel
    #path_original_excel     = os.path.join(user_paths.path_excels, "BlastoLabels.xlsx")
    path_addedID_csv        = os.path.join(user_paths.dataset, "DB_Morpheus_withID.csv")
    path_double_dish_excel  = os.path.join(user_paths.dataset, "pz con doppia dish.xlsx")

    # Percorso della directory di destinazione
    dest_dir_blastoData = user_paths.path_BlastoData


class Config_01_OpticalFlow:
    method_optical_flow = "Farneback"   # "LucasKanade/Farneback"

    # Settings
    save_metrics = True
    save_overlay_optical_flow = False
    save_final_data = False

    # Var
    img_size                    = utils.img_size
    num_minimum_frames          = 300
    num_initial_frames_to_cut   = utils.start_frame   # Taglio le prime ore perché biologicamente non significative
                                                        # (prima delle 4 ore no pronuclei e pochi movimenti cellulari)
    num_forward_frame           = 4     # Numero di frame per sum_mean_mag

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
    pyr_scale = 0.5
    levels = 4
    winSize_Farneback = 13
    iterations = 5
    poly_n = 5
    poly_sigma = 1.1

    base_out_example = f"/home/phd2/Scrivania/CorsoData/opticalFlowExamples{method_optical_flow}" 
    if method_optical_flow == "Farneback":
        output_path_optical_flow_images = f"{base_out_example}_{str(winSize_Farneback)}_{str(levels)}_{str(pyr_scale)}_{str(iterations)}_{str(poly_n)}_{str(poly_sigma)}"
    elif method_optical_flow == "LucasKanade":
                output_path_optical_flow_images = f"{base_out_example}_{str(winSize_LK)}_{str(maxLevelPyramid)}_{str(maxCorners)}"
    else:
        raise SystemExit("\n===== Scegliere un metodo di flusso ottico valido nel config =====\n")


class Config_02_temporalData:
    dict                        = "sum_mean_mag"  # mean_magnitude / sum_mean_mag / vorticity
    method_optical_flow         = "Farneback"
    type_files                  = f"files_all_days_{method_optical_flow}"
    dict_in                     = dict + "_" + method_optical_flow + ".pkl"
    
    convert_pkl_to_csv          = True
    path_pkl                    = os.path.join(type_files, dict_in)
    dictAndOptFlowType          = dict + "_" + method_optical_flow + ".csv"

    # Path in cui salvo il file csv che ottengo leggendo i pkl delle serie temporali e che sarà poi quello usato per creare il csv finale
    temporal_csv_path           = os.path.join(PROJECT_ROOT, '_02_temporalData', 'final_series_csv', dictAndOptFlowType)
    csv_file_Danilo_path        = Config_00_preprocessing.path_addedID_csv  # File che ho ottenuto dal preprocessing degli excel (singolo csv con ID)
    # Path del csv finale che contiene gli identificativi dei video, la classe e tutti i valori delle serie temporali
    final_csv_path              = os.path.join(user_paths.dataset, method_optical_flow, "FinalDataset.csv")

    embedding_type = "umap"
    use_plotly_lib = True
    path_output_dim_reduction_files = os.path.join("dim_reduction_files", method_optical_flow, dict)
    num_max_days = 7
    days_to_consider_for_dim_reduction = [1,3,5,7]    # array perché fa ciclo per poter svolgere umap su più giorni



class Config_02b_normalization:
    method_optical_flow = "Farneback"   #LucasKanade / Farneback

    # Data
    temporalDataType = Config_02_temporalData.dict
    train_size = 0.7
    embedding_type=""   # "umap" OR "tsne"
    save_normalization_example_single_pt=True
    mean_data_visualization=True
    specific_patient_to_analyse=42
    mean_data_visualization_stratified=False
    path_original_excel = user_paths.path_original_excel

    # Per gestire dati a N giorni a partire dalla i-esima ora con limiti normalizzazione
    days_to_consider = [1,3]        # Imposta il numero di giorni da considerare (1, 3, 5, o 7)
    inf_quantile = 0.05
    sup_quantile = 0.95
    initial_hours_to_cut = 3    # Remember that I already cut the first hour
    initial_frames_to_cut = initial_hours_to_cut*utils.framePerHour
    start_frame = initial_frames_to_cut+utils.start_frame

    # Paths file completo
    csv_file_path = os.path.join(user_paths.dataset, method_optical_flow, "FinalDataset.csv")

    # Base path generico per i file normalizzati
    @staticmethod
    def get_normalized_base_path(days_to_consider, method_optical_flow=method_optical_flow):
        subsets_base_path = os.path.join(user_paths.dataset, method_optical_flow,"subsets")
        return os.path.join(subsets_base_path, 
                            f"Normalized_{Config_02b_normalization.temporalDataType}_{days_to_consider}Days")


    # Metodo per ottenere i percorsi in base ai giorni selezionati
    @staticmethod
    def get_paths(days_to_consider):
        """
        Ottiene i percorsi di train, validation e test in base al numero di giorni selezionati.

        :param days_to_consider: Numero di giorni da considerare (1, 3, 5, o 7).
        :return: Tuple con i percorsi di train, validation e test.
        """
        base_path = Config_02b_normalization.get_normalized_base_path(days_to_consider=days_to_consider, 
                                                                      method_optical_flow=Config_02b_normalization.method_optical_flow)
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
    days_to_consider = 3
    project_name = utils.project_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multi_gpu = torch.cuda.device_count() > 1  # Variabile per controllare l'uso di più GPU
    num_classes = utils.num_classes
    img_size = utils.img_size
    seed = utils.seed
    num_labels = 2
    Data_shape = (1,96) #variabile di base, verrà aggiornata in ConvTran
    output_model_base_dir = os.path.join(PROJECT_ROOT, "_04_test", "best_models", method_optical_flow)
    save_plots = True
    output_dir_plots = os.path.join(PROJECT_ROOT, "_03_train", "test_results_after_training", method_optical_flow)
    path_original_excel = user_paths.path_original_excel

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
        base_path = Config_02b_normalization.get_normalized_base_path(days_to_consider=days_to_consider, 
                                                                      method_optical_flow=Config_03_train.method_optical_flow)
        train_path = f"{base_path}_train.csv"
        val_path = f"{base_path}_val.csv"
        test_path = f"{base_path}_test.csv"
        return train_path, val_path, test_path
    

    # ROCKET
    kernel_number_ROCKET     = [10000] #provato con [50,100,200,300,500,1000,5000,10000,20000]
    type_model_classification = "RF"    #or "LR" or "XGB"
    most_important_metric = "balanced_accuracy"
    
    # LSTM-FCN
    num_epochs_FCN      = 300
    batch_size_FCN      = 16
    dropout_FCN         = 0.3
    kernel_sizes_FCN    = (8,5,3) #def: 8,5,3
    filter_sizes_FCN    = (128,256,128)
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
    output_dir      = user_paths.dataset
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
    epochs          = 150        # Number of training epochs
    batch_size      = 16     # Training batch size
    lr              = 1e-3           # Learning rate
    dropout         = 0.2       # Dropout regularization ratio
    val_interval    = 1    # Evaluate on validation every XX epochs
    key_metric      = 'accuracy' # choices={'loss', 'accuracy', 'precision'}, help='Metric used for defining best epoch'
    num_classes     = utils.num_classes
    num_labels      = num_classes
    # ConvTran - Add Learning Rate Scheduler
    scheduler_patience  = 5    # Number of epochs with no improvement after which learning rate will be reduced
    scheduler_factor    = 0.5    # Factor by which the learning rate will be reduced
    # ConvTran - System
    gpu             = -1             # GPU index, -1 for CPU
    console         = False     # Optimize printout for console output; otherwise for file




class Config_03_train_with_optimization(Config_03_train):
    # Enable/disable test evaluation
    run_test_evaluation = True
    
    # Optuna optimization control
    optimize_with_optuna = True
    
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
    convtran_patience           = 60
    convtran_epochs_options     = 150
    optuna_n_trials_ConvTran    = 300