import random
import numpy as np
import torch
import torch.optim as optim


def remove_small_rows(data_list):
    # Trova l'array con la dimensione maggiore
    max_size = max(array.size for array in data_list)
    cleaned_data = [array for array in data_list if array.size == max_size]

    return cleaned_data


def create_optimizer(model, optimizer_name, lr, momentum = 0.9):
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum = momentum)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == "Adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
    elif optimizer_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == "Adamax":
        optimizer = optim.Adamax(model.parameters(), lr=lr)
    elif optimizer_name == "NAdam":
        optimizer = optim.NAdam(model.parameters(), lr=lr)
    elif optimizer_name == "ASGD":
        optimizer = optim.ASGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_name}")
    
    return optimizer


# Funzione per impostare il seed
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
