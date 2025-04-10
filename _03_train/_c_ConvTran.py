import os
import sys
from art import *
import time
import torch
import numpy as np
import logging 

# Import Project Modules
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_03_train as conf
import _utils_._utils as utils
from _c_ConvTranTraining import SupervisedTrainer, train_runner, find_best_threshold
from _c_ConvTranUtils import load_my_data
from _99_ConvTranModel.model import model_factory, count_parameters
from _99_ConvTranModel.optimizers import get_optimizer
from _99_ConvTranModel.loss import get_loss_module
from _99_ConvTranModel.utils import load_model

def Initialization():
    conf.seed_everything(conf.seed)
    device = conf.device
    print(f"Using device: {conf.device}")
    return device

def main(days_to_consider=conf.days_to_consider, 
         train_path="", val_path="", test_path="", default_path=True, 
         save_plots=conf.save_plots,
         output_dir_plots = conf.output_dir_plots,
         output_model_base_dir=conf.output_model_base_dir,
         most_important_metric = conf.most_important_metric,

         log_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "logging_files"),
         log_filename=f'train_ConvTran_based_on_{conf.method_optical_flow}'):
    
    # Makedirs
    os.makedirs(output_model_base_dir, exist_ok=True)

    # File di Log & initialization
    utils.config_logging(log_dir=log_dir, log_filename=log_filename)
    logging.info(f"\n{'='*30} Starting ConvTran Training {'='*30}")
    device = Initialization()

    if default_path:
        # Ottieni i percorsi dal config
        train_path, val_path, test_path = conf.get_paths(days_to_consider=days_to_consider)

    # Load Data
    train_loader, val_loader, test_loader = load_my_data(train_path, val_path, test_path, conf.val_ratio, conf.batch_size)

    # Aggiungi numero di etichette uniche
    conf.num_labels = len(np.unique(train_loader.dataset.labels))
    conf.Data_shape = (train_loader.dataset[0][0].shape[0], train_loader.dataset[0][0].shape[1])
    
    # Build Model
    model = model_factory(conf)
    model.to(device)

    logging.info(f"Model architecture:\n{model}")
    logging.info(f"Total parameters: {count_parameters(model):,}")

    # Optimizer and Loss
    optimizer = get_optimizer("RAdam")(model.parameters(), lr=conf.lr)
    loss_module = get_loss_module()

    # Training
    save_path = os.path.join(output_model_base_dir, f"best_convtran_model_{days_to_consider}Days.pkl")

    trainer = SupervisedTrainer(
        model, train_loader, device, loss_module, optimizer, 
        print_interval=conf.print_interval, is_training=True
    )
    val_evaluator = SupervisedTrainer(
        model, val_loader, device, loss_module,
        print_interval=conf.print_interval, is_training=False
    )

    # Start training
    logging.info("\nStarting training process...")
    train_runner(
        config=conf,
        model=model,
        trainer=trainer,
        val_evaluator=val_evaluator,
        save_path=save_path
    )
    
    # Carico il modello migliore e rivaluto sul validation set per trovare soglia ottima
    best_model, optimizer, _ = load_model(model, save_path, optimizer)
    best_model.to(device)
    best_threshold = find_best_threshold(best_model, val_loader, device)
    logging.info(f"\nOptimal threshold determined: {best_threshold:.4f}")

    # Valutazione sul test con soglia ottimale
    best_test_evaluator = SupervisedTrainer(
        best_model, test_loader, device, loss_module, 
        print_interval=conf.print_interval, is_training=False
    )

    # Esegui la valutazione e salva la matrice di confusione
    test_metrics = best_test_evaluator.evaluate_test(threshold=best_threshold)

    # Modifica del modello salvato per includere la soglia
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'best_threshold': best_threshold
    }, save_path)
    logging.info(f"\nFinal model saved to: {save_path}")

    logging.info("\n===== FINAL TEST RESULTS =====")
    for metric, value in test_metrics.items():
        if metric not in ['conf_matrix', 'fpr', 'tpr']:
            logging.info(f"{metric.capitalize()}: {value:.4f}")

    # Save plots
    if save_plots:
        complete_output_dir = os.path.join(output_dir_plots, f"day{days_to_consider}")
        os.makedirs(complete_output_dir, exist_ok=True)
        conf_matrix_filename=os.path.join(complete_output_dir,f'confusion_matrix_ConvTran_{days_to_consider}Days.png')
        utils.save_confusion_matrix(conf_matrix=test_metrics['conf_matrix'], 
                                    filename=conf_matrix_filename, 
                                    model_name="ConvTran")
        utils.plot_roc_curve(fpr=test_metrics['fpr'], tpr=test_metrics['tpr'], 
                             roc_auc=test_metrics['roc_auc'], 
                             filename=conf_matrix_filename.replace('confusion_matrix', 'roc'))

    """
    Caricamento del modello:
    model, best_threshold = load_model_with_threshold(model, save_path, device)
    """

if __name__ == '__main__':
    start_time = time.time()
    main(days_to_consider=7)
    print(f"Tempo esecuzione ConvTran: {(time.time()-start_time):.2f}s")