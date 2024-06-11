import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import wandb
import os

import sys
# Configurazione dei percorsi e dei parametri
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from networksTemporalSeries import LSTM
from config import Config_02_train as conf
from _02_train import myModelsFunctions as myFun


if __name__ == '__main__':

    # Separazione dei dati di input e delle etichette di classe
    input_data = []
    labels = []

    # Converti le liste di numpy arrays in un singolo array numpy
    input_array = np.array(input_data)
    labels_array = np.array(labels)

    # Converti l'array numpy in un tensore PyTorch
    input_tensor = torch.tensor(input_array, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_array, dtype=torch.float32)

    # Creazione di un TensorDataset
    dataset = TensorDataset(input_tensor, labels_tensor)

    # Split dei dati in train, validation, e test set
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

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

    seed_everything(conf)
    
    # Creazione dei DataLoader per train, validation e test set
    train_dataloader = DataLoader(train_dataset, batch_size = conf.batch_size, shuffle = True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size = conf.batch_size, shuffle = True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size = conf.batch_size, shuffle = True, num_workers=4)
    
    # Definisco dove far girare modello
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Definizione dei parametri della rete
    input_size = input_tensor.shape[1]      # Dimensione dell'input
    hidden_size = conf.hidden_size          # Dimensione della cella nascosta
    num_layers = conf.num_layers            # Numero di layer LSTM
    output_size = conf.output_size          # Dimensione dell'output
    bidirectional = conf.bidirectional      # Imposta a True se la rete è bidirezionale
    dropout_prob = conf.dropout_prob        # Dimensione dropout    

    # Creazione di un'istanza della rete
    model = LSTM.LSTMnetwork(input_size, hidden_size, num_layers, output_size, bidirectional, dropout_prob).to(device)

    # Definizione ottimizzatore e criterion for loss
    optimizer = myFun.create_optimizer(model, conf.optimizer_type, conf.learning_rate)     
    criterion = nn.BCEWithLogitsLoss()      # (binary cross-entropy for binary classification)

    # set wandb options
    torch.manual_seed(conf.seed)
    
    # start a new wandb run to track this script
    wandb.init(
        # Set the W&B project where this run will be logged
        project=conf.project_name,

        # Track hyperparameters and run metadata
        config={
            "exp_name": conf.exp_name,
            "dataset": conf.dataset,
            "model": conf.model_name,
            "num_epochs": conf.num_epochs,
            "batch_size": conf.batch_size,
            "learning_rate": conf.learning_rate,
            "optimizer_type": conf.optimizer_type,
            "img_size": conf.img_size, 
            "num_classes": conf.num_classes
        }
    )
    wandb.run.name = conf.exp_name
    

    for epoch in range(conf.num_epochs):
        model.train()  # Imposta la modalità di training

        epoch_train_loss = 0.0
        correct_train_predictions = 0
        total_train_samples = 0

        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()  # Azzeramento dei gradienti, altrimenti ogni volta si aggiungono a quelli calcolati al loop precedente

            # Aggiungo la dimensione del batch
            inputs = inputs.unsqueeze(1)

            # Calcolo output modello
            outputs = model(inputs)

            # Calcola la loss
            loss = criterion(torch.squeeze(outputs, 1), labels)

            # Calcola l'accuracy
            predictions = (outputs > 0.5).float()
            correct_train_predictions += (torch.squeeze(predictions, 1) == labels).sum().item()
            total_train_samples += labels.size(0)

            # Calcola i gradienti e aggiorna i pesi
            loss.backward()     # The new loss adds to what the last one computed, non lo fa la total loss
            epoch_train_loss += loss.item()
            
            optimizer.step()

        

        # Valutazione della rete su validation
        model.eval()
        with torch.no_grad():
            epoch_val_loss = 0.0
            correct_val_predictions = 0
            total_val_samples = 0
            for inputs, labels in val_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                inputs = inputs.unsqueeze(1)

                outputs = model(inputs)

                loss = criterion(torch.squeeze(outputs, 1), labels)
                epoch_val_loss += loss.item()

                predictions = (outputs > 0.5).float()
                correct_val_predictions += (torch.squeeze(predictions, 1) == labels).sum().item()
                total_val_samples += labels.size(0)


        # Calcola loss e accuracy per epoca sul train e validation set
        epoch_val_loss /= len(val_dataloader)
        epoch_train_loss /= len(train_dataloader)

        val_loss = epoch_val_loss
        val_accuracy = correct_val_predictions / total_val_samples

        train_loss = epoch_train_loss
        train_accuracy = correct_train_predictions / total_train_samples

        
        wandb.log({'epoch': epoch + 1,
                   'train_accuracy': train_accuracy,
                   'train_loss': train_loss,
                   'val_accuracy': val_accuracy,
                   'val_loss': val_loss})

        # Stampa delle informazioni sull'epoca
        print(f'Epoch [{epoch+1}/{conf.num_epochs}], Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}')




    # Valutazione finale sul test set
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        correct_test_predictions = 0
        total_test_samples = 0

        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            inputs = inputs.unsqueeze(1)

            outputs = model(inputs)

            loss = criterion(torch.squeeze(outputs, 1), labels)
            test_loss += loss.item()

            predictions = (outputs > 0.5).float()
            correct_test_predictions += (torch.squeeze(predictions, 1) == labels).sum().item()
            total_test_samples += labels.size(0)

    # Calcola loss e accuracy medie sul test set
    test_loss /= len(test_dataloader)
    test_accuracy = correct_test_predictions / total_test_samples

    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')