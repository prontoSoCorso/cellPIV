import os
import json
import torch
import numpy as np
import logging
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def Setup(config):
    output_dir = os.path.join(config.output_dir, "ConvTranResults")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    config.output_dir = output_dir
    config.save_dir = os.path.join(output_dir, 'checkpoints')
    config.pred_dir = os.path.join(output_dir, 'predictions')
    config.tensorboard_dir = os.path.join(output_dir, 'tb_summaries')

    for dir_ in [config.save_dir, config.pred_dir, config.tensorboard_dir]:
        if not os.path.exists(dir_):
            os.makedirs(dir_)

    # Converti gli attributi del config in un dizionario normale
    config_dict = {k: v for k, v in vars(config).items() if isinstance(v, (int, float, str, bool, list, dict))}
    
    with open(os.path.join(output_dir, 'configuration.json'), 'w') as fp:
        json.dump(config_dict, fp, indent=4, sort_keys=True)

    logger.info(f"Stored configuration file in '{output_dir}' as configuration.json")
    return config


def Initialization(config):
    if config.seed is not None:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
    device = torch.device('cuda' if (torch.cuda.is_available() and config.gpu != '-1') else 'cpu')
    logger.info(f"Using device: {device}")
    return device

def load_my_data(data_path, test_path, val_ratio=0.2, batch_size=16):
    data = pd.read_csv(data_path)
    labels = data.iloc[:, 2].values
    series_data = data.iloc[:, 3:].values.reshape(data.shape[0], 1, -1)

    data_test = pd.read_csv(test_path)
    y_test = data_test.iloc[:, 2].values
    X_test = data_test.iloc[:, 3:].values.reshape(data_test.shape[0], 1, -1)

    X_train, X_val, y_train, y_val = train_test_split(series_data, labels, test_size=val_ratio, random_state=42, stratify=labels)

    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    test_dataset = CustomDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)
