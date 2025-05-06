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

def main(days_to_consider=conf.days_to_consider, 
         train_path="", val_path="", test_path="", default_path=True,
         output_dir_plots = conf.output_dir_plots,
         log_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logging_files", conf.method_optical_flow),
         log_filename=f'train_ConvTran_based_on_{conf.method_optical_flow}',
         trial=None,
         run_test_evaluation=conf.run_test_evaluation,
         **kwargs):
    
    utils.config_logging(log_dir=log_dir, log_filename=log_filename)
    Initialization()
    
    if default_path:
        train_path, val_path, test_path = conf.get_paths(days_to_consider)

    if trial:  # Optuna optimization path
        params = {
            'emb_size': trial.suggest_categorical('emb_size', conf.convtran_emb_size_options),
            'num_heads': trial.suggest_categorical('num_heads', conf.convtran_num_heads_options),
            'dropout': trial.suggest_float('dropout', *conf.convtran_dropout_range),
            'batch_size': trial.suggest_categorical('batch_size', conf.convtran_batch_size_options),
            'lr': trial.suggest_float('lr', *conf.convtran_learning_rate_range, log=True)
        }
        # Only return the metric value, not the model
        _, best_metric, _ = train_convtran(params, days_to_consider, train_path, val_path, trial)
        return best_metric

    # Normal training path
    params = {
        'emb_size': kwargs.get('emb_size', conf.emb_size),
        'num_heads': kwargs.get('num_heads', conf.num_heads),
        'dropout': kwargs.get('dropout', conf.dropout),
        'batch_size': kwargs.get('batch_size', conf.batch_size),
        'lr': kwargs.get('lr', conf.lr),
        'epochs': kwargs.get('epochs', conf.epochs)
    }
    model, metrics, save_path = train_convtran(params, days_to_consider, train_path, val_path)

    if run_test_evaluation:
        test_metrics = evaluate_final_model(save_path, test_path, params['batch_size'])
        save_results(test_metrics, output_dir_plots, days_to_consider)
    
    return metrics


def train_convtran(params, days_to_consider, train_path, val_path, trial=None):
    train_loader, val_loader, _ = load_my_data(train_path, val_path, "", 
                                             conf.val_ratio, params['batch_size'])
    
    # Numero di classi e soprattutto aggiornamento dimensione del dato
    conf.num_labels = len(np.unique(train_loader.dataset.labels))
    conf.Data_shape = (train_loader.dataset[0][0].shape[0], train_loader.dataset[0][0].shape[1])
    
    # Update config with trial parameters
    conf.emb_size = params['emb_size']
    conf.num_heads = params['num_heads']
    conf.dropout = params['dropout']
    conf.batch_size = params['batch_size']
    conf.lr = params['lr']

    model = model_factory(conf).to(conf.device)
    optimizer = get_optimizer("RAdam")(model.parameters(), lr=conf.lr)
    loss_module = get_loss_module()

    # print_interval=conf.print_interval, is_training=False ?????
    trainer = SupervisedTrainer(model, train_loader, conf.device, loss_module, optimizer)
    val_evaluator = SupervisedTrainer(model, val_loader, conf.device, loss_module)
    save_path_tmp=os.path.join(conf.output_model_base_dir, f"best_convtran_model_{days_to_consider}Days.pkl")

    best_metric, best_threshold = train_runner(
        config=conf, model=model, trainer=trainer, val_evaluator=val_evaluator, 
        save_path=save_path_tmp,
        trial=trial
    )

    return model, best_metric, save_path_tmp

def evaluate_final_model(save_path, test_path, batch_size):
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
    _, _, test_loader = load_my_data("", "", test_path, 0, batch_size)
    evaluator = SupervisedTrainer(model, test_loader, conf.device, get_loss_module())
    return evaluator.evaluate_test(threshold=best_threshold)

def save_results(metrics, output_dir, days):
    logging.info("\n===== FINAL TEST RESULTS =====")
    for metric, value in metrics.items():
        if metric not in ['conf_matrix', 'fpr', 'tpr']:
            logging.info(f"{metric.capitalize()}: {value:.4f}")
    
    if conf.save_plots:
        complete_output_dir = os.path.join(output_dir, f"day{days}")
        os.makedirs(complete_output_dir, exist_ok=True)
        conf_matrix_filename=os.path.join(complete_output_dir,f'confusion_matrix_ConvTran_{days}Days.png')
        utils.save_confusion_matrix(conf_matrix=metrics['conf_matrix'], 
                                    filename=conf_matrix_filename, 
                                    model_name="ConvTran")
        utils.plot_roc_curve(fpr=metrics['fpr'], tpr=metrics['tpr'],
                             roc_auc=metrics['roc_auc'], 
                             filename=conf_matrix_filename.replace('confusion_matrix', 'roc'))


if __name__ == '__main__':
    start_time = time.time()
    main(days_to_consider=7)
    print(f"Tempo esecuzione ConvTran: {(time.time()-start_time):.2f}s")