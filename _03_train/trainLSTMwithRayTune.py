import os
import pickle
import numpy as np
import ray
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import matplotlib.pyplot as plt
import wandb
import random
from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback

import os
# Ottieni il percorso del file corrente
current_file_path = os.path.abspath(__file__)
# Risali la gerarchia fino alla cartella "cellPIV"
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
import sys
sys.path.append(parent_dir)

from networksTemporalSeries import LSTM
from config import Config_03_train_rocket as conf
from _02_train import myModelsFunctions as myFun





def load_pickled_files(directory):
    files = os.listdir(directory)
    data = {}
    for file in files:
        if file.endswith(".pkl"):
            with open(os.path.join(directory, file), 'rb') as f:
                data[file[:-4]] = pickle.load(f)
    return data


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



def train_model(config):

    # Definire il percorso della cartella dove sono stati salvati i file
    data_path = conf.data_path
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path {data_path} does not exist")

    # Caricare i file dalla cartella specificata
    loaded_data = load_pickled_files(data_path)
    if "sum_mean_mag_mat" not in loaded_data:
        raise KeyError("Key 'sum_mean_mag_mat' not found in loaded data")

    print(loaded_data.keys())  # Mostro i nomi dei file caricati

    # Estrazione dei dati relativi a sum_mean_mag
    sum_mean_mag_data = loaded_data["sum_mean_mag_mat"]

    # Rimuovo tutti gli array nulli
    sum_mean_mag_data = myFun.remove_small_rows(sum_mean_mag_data)

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

    # Split dei dati in train, validation, e test set
    train_size = int(conf.train_size * len(dataset))
    val_size = len(dataset) - train_size 
    train_dataset, val_dataset= torch.utils.data.random_split(dataset, [train_size, val_size])

    # Creazione dei DataLoader per train, validation e test set
    train_dataloader = DataLoader(train_dataset, batch_size = conf.batch_size, shuffle = True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size = conf.batch_size, shuffle = True, num_workers=4)
    
    # Definisco dove far girare modello
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Definizione dei parametri della rete
    input_size = input_tensor.shape[1]      # Dimensione dell'input
    hidden_size = config['hidden_size']          # Dimensione della cella nascosta
    num_layers = config['num_layers']            # Numero di layer LSTM
    output_size = conf.output_size          # Dimensione dell'output
    bidirectional = config['bidirectional']      # Imposta a True se la rete è bidirezionale
    dropout_prob = config['dropout_prob']        # Dimensione dropout    

    # Creazione di un'istanza della rete
    model = LSTM.LSTMnetwork(input_size, hidden_size, num_layers, output_size, bidirectional, dropout_prob).to(device)

    # Definizione ottimizzatore e criterion for loss
    optimizer = myFun.create_optimizer(model, config['optimizer_type'], config['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()      # (binary cross-entropy for binary classification)


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

        tune.report(train_loss=train_loss, train_accuracy=train_accuracy, val_loss=val_loss, val_accuracy=val_accuracy)

        # Stampa delle informazioni sull'epoca
        print(f'Epoch [{epoch+1}/{conf.num_epochs}], Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}')






if __name__ == '__main__':
    # Configuration of ray tmp dir
    os.environ["RAY_TMPDIR"] = "C:/Utenti/loren/AppData/Local/Temp/ray/"
    ray.init(_temp_dir="C:/Utenti/loren/AppData/Local/Temp/ray/")
    
    seed_everything(conf.seed)

    search_space = {
        'num_epochs': tune.grid_search([10]),
        'batch_size': tune.grid_search([16, 32]),
        'learning_rate': tune.grid_search([0.001, 0.0005]),
        'hidden_size': tune.grid_search([64, 128]),
        'num_layers': tune.grid_search([2, 3, 5]),
        'bidirectional': tune.grid_search([True]),
        'dropout_prob': tune.grid_search([0.2, 0.5]),
        'optimizer_type': tune.grid_search(['Adam', 'RMSprop'])
    }

    analysis = tune.run(
        train_model,  # La funzione da eseguire per ogni configurazione
        config=search_space,  # Il dizionario contenente lo spazio di ricerca dei parametri
        resources_per_trial={'cpu': 4, 'gpu': 1 if torch.cuda.is_available() else 0},  # Risorse assegnate per ogni esecuzione
        num_samples=1,  # Numero di campioni per ogni configurazione (qui è 1 perché usiamo la grid search)
        scheduler=None,  # Non utilizziamo uno scheduler specifico in questo caso
        progress_reporter=tune.CLIReporter(),  # Reporter per visualizzare l'andamento della grid search nella console
        local_dir=conf.local_dir,  # Directory locale dove salvare i risultati degli esperimenti
        callbacks=[WandbLoggerCallback(
            project=conf.project_name,  # Nome del progetto su WandB
            api_key_file=conf.keyAPIpath,  # Percorso al file contenente la chiave API di WandB
            log_config=True  # Logga anche la configurazione dei parametri su WandB
        )]
    )


    print("Best hyperparameters found were: ", analysis.best_config)











'''
PER UN EVENTUALE SCHEDULER
ASHA: uno dei più efficienti e comunemente utilizzati, ideale per la maggior parte delle applicazioni.
Median Stopping:  semplice e veloce da configurare
PBT: molto potente ma più complesso da configurare.



from ray.tune.schedulers import ASHAScheduler

scheduler = ASHAScheduler(
    metric="val_loss",  # La metrica da monitorare
    mode="min",  # "min" per minimizzare la metrica, "max" per massimizzarla
    max_t=100,  # Tempo massimo per ogni trial (in epoche, step, ecc.)
    grace_period=1,  # Numero minimo di step prima di considerare l'interruzione di un trial
    reduction_factor=2  # Fattore di riduzione
)

analysis = tune.run(
    train_model,
    config=search_space,
    resources_per_trial={'cpu': 4, 'gpu': 1 if torch.cuda.is_available() else 0},
    num_samples=10,
    scheduler=scheduler,
    progress_reporter=tune.CLIReporter(),
    local_dir="ray_results",
    callbacks=[WandbLoggerCallback(
        project=conf.project_name,
        api_key_file="C:/Users/loren/wandb_api_key.txt",
        log_config=True
    )]
)



'''


