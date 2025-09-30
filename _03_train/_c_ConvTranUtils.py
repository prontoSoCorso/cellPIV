import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch

def load_my_data(X_train=None, y_train=None, 
                 X_val=None, y_val=None, 
                 X_test=None, y_test=None, 
                 batch_size=16):
    
    # Initialize loaders as None
    train_loader, val_loader, test_loader = None, None, None
    
    # Create datasets and loaders if data is provided
    if X_train is not None and y_train is not None:
        print("Loading training data ...")
        train_dataset = CustomDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if X_val is not None and y_val is not None:
        print("Loading validation data ...")
        val_dataset = CustomDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if X_test is not None and y_test is not None:
        print("Loading test data ...")
        test_dataset = CustomDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def load_model(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint.get('best_threshold', 0.5)


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)
    