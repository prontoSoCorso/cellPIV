import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import matplotlib.pyplot as plt
import wandb


import sys
sys.path.append("C:/Users/loren/OneDrive - Università di Pavia/Magistrale - Sanità Digitale/Tesi Magistrale/cellPIV")
from networksTemporalSeries import LSTM
from config import Config_02_Model as conf



def load_pickled_files(directory):
    files = os.listdir(directory)
    data = {}
    for file in files:
        if file.endswith(".pkl"):
            with open(os.path.join(directory, file), 'rb') as f:
                data[file[:-4]] = pickle.load(f)
    return data



if __name__ == '__main__':

    # Definire il percorso della cartella dove sono stati salvati i file
    data_path = conf.data_path

    # Caricare i file dalla cartella specificata
    loaded_data = load_pickled_files(data_path)

    print(loaded_data.keys())  # Mostro i nomi dei file caricati

    # Estrazione dei dati relativi a sum_mean_mag
    sum_mean_mag_data = loaded_data["sum_mean_mag_mat"]

    # Separazione dei dati di input e delle etichette di classe
    input_data = []
    labels = []

    for sequence in sum_mean_mag_data:
        input_sequence = sequence[:-1]  # Rimuovi l'ultimo elemento (classe)
        label = sequence[-1]  # Ultimo elemento (classe)
        input_data.append(input_sequence)
        labels.append(label)

    # Converti le liste di numpy arrays in un singolo array numpy
    input_array = np.array(input_data)
    labels_array = np.array(labels)

    # Converti l'array numpy in un tensore PyTorch
    input_tensor = torch.tensor(input_array, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_array, dtype=torch.float32)

    # Creazione di un TensorDataset
    dataset = TensorDataset(input_tensor, labels_tensor)

    # Definizione dei batch size (numero di sequenze prese)
    batch_size = 3

    # Split dei dati in train, validation, e test set
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # Creazione dei DataLoader
    # Creazione dei DataLoader per train, validation e test set
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    # Definisco dove far girare modello
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Definizione dei parametri della rete
    input_size = input_tensor.shape[1]  # Dimensione dell'input
    hidden_size = 64                    # Dimensione della cella nascosta
    num_layers = 2                      # Numero di layer LSTM
    output_size = 1                     # Dimensione dell'output
    bidirectional = False               # Imposta a True se la rete è bidirezionale

    # Creazione di un'istanza della rete
    model = LSTM.LSTMnetwork(input_size, hidden_size, num_layers, output_size, bidirectional).to(device)
    # Definisci l'ottimizzatore
    learning_rate = 0.001  # Puoi regolare il tasso di apprendimento secondo necessità
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define the criterion for calculating loss (binary cross-entropy for binary classification)
    criterion = nn.BCEWithLogitsLoss()

    # Numero di epoche
    num_epochs = conf.epochs

    # Inizializzazione di liste per loss accuracy durante le epoche (per train e validation)
    train_losses = []
    train_accuracies = []

    val_losses = []
    val_accuracies = []

    # start a new wandb run to track this script
    prova_push = "riga aggiunta da desktop - 15:45"
    wandb.init(
        # set the wandb project where this run will be logged
        project = conf.project_name,

        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.02,
        "architecture": "LSTM",
        "dataset": "Blasto",
        "epochs": conf.epochs,
        }
    )

    wandb.init(project=project_name, 
               config={"exp_name": exp_name, "model": model_name,
                       "config": config_name,
                       "pretrain": pretrain, "learning_rate": lr,
                       "dropout": dropout, "l2": l2,
                       "batch_size": batch_size,
                       "epochs": epochs, "milestones": milestones,
                       "lsmooth": lsmooth, "pos_weight": pos_weight,
                       "img_size": img_size, "num_classes": num_classes, 
                       "num_frames": num_frames})
    wandb.run.name = exp_name



    for epoch in range(num_epochs):
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

        # Calcola loss e accuracy medie per epoca
        epoch_train_loss /= len(train_dataloader)
        train_loss = epoch_train_loss
        train_accuracy = correct_train_predictions / total_train_samples
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

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

        # Calcola loss e accuracy medie per epoca sul validation set
        epoch_val_loss /= len(val_dataloader)
        val_loss = epoch_val_loss
        val_accuracy = correct_val_predictions / total_val_samples
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Stampa delle informazioni sull'epoca
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}')



    # Plot loss e accuracy
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.legend()

    plt.show()



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






# wandb.ai
'''
Installo 
faccio wandb login e scrivo API che mi dice dal sito
Importo come libreria (import wandb)


wandb.init(project = project_name) per fare dizionario per campi da vedere su wandb
wandb.run.name = exp_name (magari concateno tutti i parametri)


quando voglio fare il grafico faccio
wandb.log({})
e ci metto dentro tutto quello che voglio graficare




'''

