''' Configuration file for the project'''

import torch



class Config_01_OpticalFlow:
    project_name = 'BlastoClass_y13-18_3days_288frames_optflow_LK'
    data_path = 'C:/Users/loren/OneDrive - Università di Pavia/Magistrale - Sanità Digitale/Tesi Magistrale/cellPIV/_01_opticalFlows'
    seed = 2024

    img_size=500
    num_frames=288



class Config_02_Model:
    project_name = 'BlastoClass_y13-18_3days_288frames_optflow_LK'
    data_path = 'C:/Users/loren/OneDrive - Università di Pavia/Magistrale - Sanità Digitale/Tesi Magistrale/cellPIV/_02_Model'
    model_name = 'LSTM'
    seed = 2024

    epochs = 100
    batch_size = 2
    pos_weight = torch.tensor(1)
    img_size=500
    num_classes=2
    num_frames=288


class 









