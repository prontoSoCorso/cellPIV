import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch
import seaborn as sns

# Metodo per plottare curva ROC
def plot_roc_curve(fpr, tpr, roc_auc, filename):
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()


# Metodo per salvare la matrice di confusione
def save_confusion_matrix(conf_matrix, filename):
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False, xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - ConvTran")
    plt.savefig(filename)
    plt.close()


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
    