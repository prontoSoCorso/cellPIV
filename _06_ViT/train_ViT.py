import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import h5py
import os
from model_ViT import VisionTransformer
from config_ViT import *
from tqdm import tqdm

class EmbryoDataset(Dataset):
    def __init__(self, h5_path, split):
        self.h5_path = h5_path
        with h5py.File(h5_path, 'r') as hf:
            self.split = split
            self.indices = [i for i in range(len(hf['split'])) if hf['split'][i].decode() == split]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as hf:
            actual_idx = self.indices[idx]
            image = hf['images'][actual_idx]
            label = hf['classes'][actual_idx]
        return torch.tensor(image).unsqueeze(0).float(), torch.tensor(label).long()

def train():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    train_dataset = EmbryoDataset(DATA_PATH, 'train')
    val_dataset = EmbryoDataset(DATA_PATH, 'val')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = VisionTransformer().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Val Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_model.pth"))
            print(f"New best model saved with accuracy {acc:.2f}%")

if __name__ == "__main__":
    train()