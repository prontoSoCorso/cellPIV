import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def load_my_data(train_path, val_path, test_path, val_ratio=0.2, batch_size=16):
    print("Reading training data ...")
    data_train = pd.read_csv(train_path)
    X_train = data_train.iloc[:, 3:].values.reshape(data_train.shape[0], 1, -1)
    y_train = data_train["BLASTO NY"].values

    print("Reading validation data ...")
    data_val = pd.read_csv(val_path)
    X_val = data_val.iloc[:, 3:].values.reshape(data_val.shape[0], 1, -1)
    y_val = data_val["BLASTO NY"].values

    print("Reading test data ...")
    data_test = pd.read_csv(test_path)
    X_test = data_test.iloc[:, 3:].values.reshape(data_test.shape[0], 1, -1)
    y_test = data_test["BLASTO NY"].values

    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    test_dataset = CustomDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Data loaders ready.")
    return train_loader, val_loader, test_loader

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)
    