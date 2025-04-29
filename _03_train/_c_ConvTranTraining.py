import torch
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import logging
import optuna

import _utils_._utils as utils

class SupervisedTrainer:
    def __init__(self, model, data_loader, device, criterion, optimizer=None, print_interval=100, writer=None, is_training=True):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.print_interval = print_interval
        self.writer = writer
        self.is_training = is_training
        self.total_correct = 0
        self.total_samples = 0

    def train_epoch(self, epoch):
        self.model.train()
        total_correct = 0
        total_samples = 0

        for i, (inputs, targets) in enumerate(self.data_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            if loss.dim() > 0:
                loss = loss.mean()

            loss.backward()
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Calcolo dell'accuratezza
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(targets).sum().item()
            total_samples += targets.size(0)

            """
            # Stampa la perdita e l'accuratezza nel terminale
            if i % self.print_interval == 0:
                logging.info(f'Train Step: {i}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')
            """

    def evaluate(self, epoch=None, keep_all=True):
        self.model.eval()
        all_metrics = []
        self.total_correct = 0
        self.total_samples = 0

        targets_list = []
        preds_list = []

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Mi assicuro che la perdita sia scalare
                if loss.dim() > 0:
                    loss = loss.mean()
                
                if keep_all:
                    all_metrics.append(loss.item())

                # Calcolo dell'accuratezza
                _, predicted = outputs.max(1)
                self.total_correct += predicted.eq(targets).sum().item()
                self.total_samples += targets.size(0)

                # Collect all predictions and true targets for additional metrics
                preds_list.extend(predicted.cpu().numpy())
                targets_list.extend(targets.cpu().numpy())
                
            # Calcolo delle metriche
            accuracy = self.total_correct / self.total_samples if self.total_samples > 0 else 0
            
        # Evito divisione per zero
        if len(all_metrics) == 0:
            return float('inf'), all_metrics  # Restituisco infinito se non ci sono metriche

        val_loss = sum(all_metrics) / len(all_metrics)
        return  val_loss, accuracy
    

    def get_accuracy(self):
        return self.total_correct / self.total_samples if self.total_samples > 0 else 0


    # Metodo per la valutazione finale sul test (include metriche avanzate e salvataggio della matrice di confusione)
    def evaluate_test(self, threshold=0.5):
        self.model.eval()
        targets_list = []
        preds_list = []
        probs_list = []

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                # Calcolo probabilità per ROC e predizione con threshold
                probs = torch.softmax(outputs, dim=1)[:, 1]  # Probabilità per la classe positiva
                preds = (probs >= threshold).int()

                # Salva predizioni e target
                probs_list.extend(probs.cpu().numpy())
                preds_list.extend(preds.cpu().numpy())
                targets_list.extend(targets.cpu().numpy())

        # Calcolo metriche complete
        metrics = utils.calculate_metrics(y_true=targets_list, y_pred=preds_list, y_prob=probs_list)
        return metrics
    

def find_best_threshold(model, val_loader=None, y_true=None, y_probs=None):
    """ Trova la soglia ottimale sul validation set """
    if model:
        model.eval()
        y_true, y_prob = [], []
        
        with torch.no_grad():
            for X, y in val_loader:
                outputs = model(X)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                y_prob.extend(probs.cpu().numpy())
                y_true.extend(y.cpu().numpy())
    
    thresholds = np.linspace(0.0, 1.0, 101)
    best_threshold = 0.5
    best_metric = 0.0
    for th in thresholds:
        current_metric = balanced_accuracy_score(y_true, (np.array(y_probs) >= th))
        
        if current_metric > best_metric:
            best_metric = current_metric
            best_threshold = th

    return best_threshold


def train_runner(config, model, trainer, val_evaluator, save_path, trial=None):
    best_val_metric = float('-inf')
    epochs_no_improve = 0
    early_stopping_delta = 0.001
    best_threshold = 0.5

    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizer, mode='min', factor=config.scheduler_factor, patience=config.scheduler_patience)

    for epoch in range(config.epochs):  # Config non è dizionario, uso notazione a punto per accedere agli attributi
        # Training phase
        trainer.train_epoch(epoch)

        # Validation phase
        val_true, val_probs = [], []
        with torch.no_grad():
            for X, y in val_evaluator.data_loader:
                X, y = X.to(config.device), y.to(config.device)
                outputs = model(X)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                val_probs.extend(probs.cpu().numpy())
                val_true.extend(y.cpu().numpy())
        
        # Find best threshold for current epoch
        current_threshold = find_best_threshold(model=None, y_true=val_true, y_probs=val_probs)
        current_metric = balanced_accuracy_score(val_true, (np.array(val_probs) >= current_threshold))
        scheduler.step(current_metric)
        if (epoch + 1) % 10 == 0:
            logging.info(f"Validation Metric - Epoch [{epoch+1}/{config.epochs}]: {current_metric:.4f}")
        
        # Report intermediate results for pruning
        if trial:
            trial.report(best_val_metric, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Early stopping logic and Update best metric and threshold
        if current_metric > (best_val_metric+ early_stopping_delta):
            best_val_metric = current_metric
            best_threshold = current_threshold
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_threshold': best_threshold
            }, save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.convtran_patience:
                break

    return best_val_metric, best_threshold