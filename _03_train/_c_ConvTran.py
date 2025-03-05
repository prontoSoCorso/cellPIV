import os
import sys
from art import *
import time
import torch
import numpy as np

# Import Project Modules
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_03_train as config
from _c_ConvTranTraining import SupervisedTrainer, train_runner, find_best_threshold
from _c_ConvTranUtils import load_my_data, save_confusion_matrix, plot_roc_curve
from _99_ConvTranModel.model import model_factory, count_parameters
from _99_ConvTranModel.optimizers import get_optimizer
from _99_ConvTranModel.loss import get_loss_module
from _99_ConvTranModel.utils import load_model

def Initialization():
    config.seed_everything(config.seed)
    device = config.device
    print(f"Using device: {config.device}")
    return device

def main(days_to_consider=config.days_to_consider, 
         save_conf_matrix=True,
         output_dir_plots = parent_dir):
    
    device = Initialization()

    # Ottieni i percorsi dal config
    train_path, val_path, test_path = config.get_paths(days_to_consider)

    # Load Data
    train_loader, val_loader, test_loader = load_my_data(train_path, val_path, test_path, config.val_ratio, config.batch_size)

    # Aggiungi numero di etichette uniche
    config.num_labels = len(np.unique(train_loader.dataset.labels))
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
    save_path = os.path.join(parent_dir, config.test_dir, f"best_convTran_model_{days_to_consider}Days.pkl")

    trainer = SupervisedTrainer(
        model, train_loader, device, loss_module, optimizer, 
        print_interval=config.print_interval, is_training=True
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
    
    # Carico il modello migliore e rivaluto sul validation set per trovare soglia ottima
    best_model, optimizer, _ = load_model(model, save_path, optimizer)
    best_model.to(device)
    best_threshold = find_best_threshold(best_model, val_loader, device)
    print(f"\nBest Threshold Found: {best_threshold:.4f}")

    # Valutazione sul test con soglia ottimale
    best_test_evaluator = SupervisedTrainer(
        best_model, test_loader, device, loss_module, 
        print_interval=config.print_interval, is_training=False
    )

    # Esegui la valutazione e salva la matrice di confusione
    test_metrics = best_test_evaluator.evaluate_test(threshold=best_threshold)

    # Modifica del modello salvato per includere la soglia
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'best_threshold': best_threshold
    }, save_path)

    print("\n===== FINAL TEST RESULTS - ConvTran =====")
    for metric, value in test_metrics.items():
        if metric not in ['conf_matrix', 'fpr', 'tpr']:
            print(f"{metric.capitalize()}: {value:.4f}")
    plot_roc_curve(test_metrics['fpr'], test_metrics['tpr'], 
                   test_metrics['roc_auc'], 
                   os.path.join(output_dir_plots, f"roc_curve_LSTMFCN_{days_to_consider}Days.png"))

    # Salvataggio output
    conf_matrix_filename=os.path.join(output_dir_plots,f'confusion_matrix_ConvTran_{days_to_consider}Days.png')
    if save_conf_matrix:
        save_confusion_matrix(test_metrics['conf_matrix'], conf_matrix_filename)
        plot_roc_curve(test_metrics['fpr'], test_metrics['tpr'], test_metrics['roc_auc'], 
                       conf_matrix_filename.replace('confusion', 'roc'))


    """
    Caricamento del modello:
    model, best_threshold = load_model_with_threshold(model, save_path, device)
    """

if __name__ == '__main__':
    start_time = time.time()
    main(days_to_consider=7)
    print(f"Tempo esecuzione ConvTran: {(time.time()-start_time):.2f}s")