import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch

def load_my_data(train_path, val_path, test_path, val_ratio=0.2, batch_size=16):
    train_loader = None
    val_loader = None
    test_loader = None

    if train_path:
        print("Reading training data ...")
        data_train = pd.read_csv(train_path)
        temporal_columns = [col for col in data_train.columns if col.startswith('value_')]  # Equal for train, val e test
        X_train = data_train[temporal_columns].values.reshape(data_train.shape[0], 1, -1)
        y_train = data_train["BLASTO NY"].values

        print("Reading validation data ...")
        data_val = pd.read_csv(val_path)
        X_val = data_val[temporal_columns].values.reshape(data_val.shape[0], 1, -1)
        y_val = data_val["BLASTO NY"].values

        train_dataset = CustomDataset(X_train, y_train)
        val_dataset = CustomDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if test_path:
        print("Reading test data ...")
        data_test = pd.read_csv(test_path)
        temporal_columns = [col for col in data_test.columns if col.startswith('value_')]
        X_test = data_test[temporal_columns].values.reshape(data_test.shape[0], 1, -1)
        y_test = data_test["BLASTO NY"].values
        test_dataset = CustomDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
    print("Data loaders ready.")

    return train_loader, val_loader, test_loader


def load_model_with_threshold(model, path, device):
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
    