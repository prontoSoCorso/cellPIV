import os
import sys
from art import *
import torch

# Import Project Modules
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from ConvTranUtils import load_my_data, CustomDataset
from _99_ConvTranModel.model import model_factory, count_parameters
from _99_ConvTranModel.optimizers import get_optimizer
from _99_ConvTranModel.loss import get_loss_module
from _99_ConvTranModel.utils import load_model
from ConvTranTraining import SupervisedTrainer, train_runner
from config import Config_03_train as config

def Initialization():
    config.seed_everything(config.seed)
    device = config.device
    print(f"Using device: {config.device}")
    return device

if __name__ == '__main__':
    device = Initialization()

    # Specifica il numero di giorni da considerare
    days_to_consider = 1

    # Ottieni i percorsi dal config
    train_path, val_path, test_path = config.get_paths(days_to_consider)

    # Load Data
    train_loader, val_loader, test_loader = load_my_data(train_path, val_path, test_path, config.val_ratio, config.batch_size)

    # Aggiungi numero di etichette uniche
    config.num_labels = len(set(train_loader.dataset.labels))
    config.Data_shape = (train_loader.dataset[0][0].shape[0], train_loader.dataset[0][0].shape[1])
    
    # Build Model
    model = model_factory(config)
    model.to(device)

    print(f"Model:\n{model}")
    print(f"Total number of parameters: {count_parameters(model)}")

    # Optimizer and Loss
    optimizer = get_optimizer("RAdam")(model.parameters(), lr=config.lr)
    loss_module = get_loss_module()

    # Training
    save_path = os.path.join(parent_dir, config.test_dir, "best_convTran_model_" + str(days_to_consider) + "Days.pkl")

    trainer = SupervisedTrainer(
        model, train_loader, device, loss_module, optimizer, 
        print_interval=config.print_interval
    )
    val_evaluator = SupervisedTrainer(
        model, val_loader, device, loss_module, 
        print_interval=config.print_interval, is_training=False
    )

    train_runner(
        config=config,
        model=model,
        trainer=trainer,
        val_evaluator=val_evaluator,
        save_path=save_path
    )
    
    # Carica il modello migliore e valuta sul test set
    best_model, optimizer, start_epoch = load_model(model, save_path, optimizer)
    best_model.to(device)

    best_test_evaluator = SupervisedTrainer(
        best_model, test_loader, device, loss_module, 
        print_interval=config.print_interval, is_training=False
    )

    # Esegui la valutazione e salva la matrice di confusione
    test_metrics = best_test_evaluator.evaluate_test(
        save_conf_matrix=True,
        conf_matrix_filename='confusion_matrix_ConvTran_' + str(days_to_consider) + 'Days.png'
    )
    print(f"Best Model Test Metrics: {test_metrics}")