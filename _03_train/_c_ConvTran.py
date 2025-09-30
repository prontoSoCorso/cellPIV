import os
import sys
import time
import torch
import numpy as np
import logging
import optuna

# Import Project Modules
parent_dir = os.path.dirname(os.path.abspath(__file__))
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_03_train_with_optimization as conf
import _utils_._utils as utils
from _c_ConvTranUtils import load_my_data
from _c_ConvTranTraining import SupervisedTrainer, train_runner
from _99_ConvTranModel.model import model_factory
from _99_ConvTranModel.optimizers import get_optimizer
from _99_ConvTranModel.loss import get_loss_module
from _99_ConvTranModel.utils import load_model

device = conf.device

def Initialization():
    conf.seed_everything(conf.seed)
    device = conf.device
    print(f"Using device: {conf.device}")
    return device


def train_convtran(params, day_label, trial=None, 
                   X_train=None, y_train=None, 
                   X_val=None, y_val=None):
    # Load dataloaders
    train_loader, val_loader, _ = load_my_data(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, batch_size=params['batch_size'])

    # Numero di classi e soprattutto aggiornamento dimensione del dato
    conf.num_labels = len(np.unique(train_loader.dataset.labels))
    conf.Data_shape = (train_loader.dataset[0][0].shape[0], train_loader.dataset[0][0].shape[1])
    
    # Update config with trial parameters
    conf.emb_size_convtran = params['emb_size']
    conf.num_heads_convtran = params['num_heads']
    conf.dropout_convtran = params['dropout']
    conf.batch_size_convtran = params['batch_size']
    conf.lr_convtran = params['lr']

    # Build model, optimizer, and loss function
    model = model_factory(conf).to(conf.device)
    print("Model architecture:")
    print(model)
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    optimizer = get_optimizer("RAdam")(model.parameters(), lr=conf.lr_convtran)
    loss_module = get_loss_module()

    # Initialize trainer and evaluator
    trainer = SupervisedTrainer(model, train_loader, conf.device, loss_module, optimizer)
    val_evaluator = SupervisedTrainer(model, val_loader, conf.device, loss_module)
    save_path_tmp=os.path.join(conf.output_model_base_dir, f"best_convtran_model_{day_label}Days.pkl")

    # Train the model
    model, best_metric, best_threshold = train_runner(
        config=conf, model=model, trainer=trainer, val_evaluator=val_evaluator, 
        save_path=save_path_tmp,
        trial=trial
    )

    return model, best_metric, save_path_tmp

def evaluate_final_model(save_path, X_test, y_test, batch_size):
    # Load model AND threshold
    checkpoint = torch.load(save_path, map_location=conf.device, weights_only=False)

    # Restore parameters to config
    saved_config = checkpoint['config']
    for key, value in saved_config.items():
        setattr(conf, key, value)

    # Rebuild model with original config and save best threshold
    model = model_factory(conf).to(conf.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    best_threshold = checkpoint['best_threshold']

    # Evaluate with loaded threshold
    _, _, test_loader = load_my_data(X_test=X_test, y_test=y_test, batch_size=batch_size)
    evaluator = SupervisedTrainer(model, test_loader, conf.device, get_loss_module())
    return evaluator.evaluate_test(threshold=best_threshold)


def main(data,
         day_label=conf.days_label, 
         output_dir_plots = conf.output_dir_plots,
         log_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logging_files", conf.method_optical_flow),
         log_filename=f'train_ConvTran_based_on_{conf.method_optical_flow}',
         trial=None,
         run_test_evaluation=conf.run_test_evaluation,
         **kwargs):
    
    utils.config_logging(log_dir=log_dir, log_filename=log_filename)
    Initialization()
    logging.info(f"=== Training ConvTran for day {day_label} ===")
    # Unpack & validate
    X_train, y_train, X_val, y_val, X_test, y_test = utils._check_data_dict(
        data, require_test=bool(run_test_evaluation)
    )

    if trial:  # Optuna optimization path
        params = {
            'emb_size': trial.suggest_categorical('emb_size', conf.convtran_emb_size_options),
            'num_heads': trial.suggest_categorical('num_heads', conf.convtran_num_heads_options),
            'dropout': trial.suggest_float('dropout', *conf.convtran_dropout_range),
            'batch_size': trial.suggest_categorical('batch_size', conf.convtran_batch_size_options),
            'lr': trial.suggest_float('lr', *conf.convtran_learning_rate_range, log=True)
        }
        # Only return the metric value, not the model
        _, best_metric, _ = train_convtran(params=params, day_label=day_label, trial=trial, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
        return best_metric

    # Normal training path
    params = {
        'emb_size': kwargs.get('emb_size', conf.emb_size_convtran),
        'num_heads': kwargs.get('num_heads', conf.num_heads_convtran),
        'dropout': kwargs.get('dropout', conf.dropout_convtran),
        'batch_size': kwargs.get('batch_size', conf.batch_size_convtran),
        'lr': kwargs.get('lr', conf.lr_convtran),
        'epochs': kwargs.get('epochs', conf.epochs_convtran)
    }
    model, metrics, save_path = train_convtran(params=params, day_label=day_label, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    if run_test_evaluation:
        test_metrics = evaluate_final_model(save_path=save_path, X_test=X_test, y_test=y_test, batch_size=params['batch_size'])
        utils.save_results(test_metrics, output_dir_plots, "ConvTran", day_label)
    
    return metrics


if __name__ == '__main__':
    start_time = time.time()
    main(day_label=7)
    print(f"Tempo esecuzione ConvTran: {(time.time()-start_time):.2f}s")