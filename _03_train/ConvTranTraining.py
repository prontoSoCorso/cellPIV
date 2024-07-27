import torch
import logging

logger = logging.getLogger(__name__)

class SupervisedTrainer:
    def __init__(self, model, data_loader, device, criterion, optimizer=None, print_interval=100):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.print_interval = print_interval

    def train_epoch(self):
        self.model.train()
        for i, (inputs, targets) in enumerate(self.data_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Mi assicuro che la perdita sia scalare
            if loss.dim() > 0:  # Se la perdita ha più di 0 dimensioni, riduco a uno scalare
                loss = loss.mean()

            loss.backward()
            self.optimizer.step()

            if i % self.print_interval == 0:
                logger.info(f'Train Step: {i}, Loss: {loss.item()}')

    def evaluate(self, keep_all=False):
        self.model.eval()
        all_metrics = []
        with torch.no_grad():
            for inputs, targets in self.data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Mi assicuro che la perdita sia scalare
                if loss.dim() > 0:
                    loss = loss.mean()
                
                if keep_all:
                    all_metrics.append(loss.item())

        # Evito divisione per zero
        if len(all_metrics) == 0:
            return float('inf'), all_metrics  # Restituisco infinito se non ci sono metriche

        return sum(all_metrics) / len(all_metrics), all_metrics

def train_runner(config, model, trainer, val_evaluator, save_path):
    best_val_loss = float('inf')
    for epoch in range(config.epochs):  # Config non è dizionario, uso notazione a punto per accedere agli attributi
        logger.info(f'Epoch {epoch+1}/{config.epochs}')
        trainer.train_epoch()
        if epoch % config.val_interval == 0:
            val_loss, _ = val_evaluator.evaluate(keep_all = True)
            logger.info(f'Validation Loss: {val_loss}')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                logger.info(f'Saved best model at epoch {epoch+1}')
