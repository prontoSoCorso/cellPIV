import torch
import logging

logger = logging.getLogger(__name__)

class SupervisedTrainer:
    def __init__(self, model, data_loader, device, criterion, optimizer=None, print_interval=100, writer=None, is_training=True):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.print_interval = print_interval
        self.writer = writer  # SummaryWriter per TensorBoard
        self.is_training = is_training

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
            self.optimizer.step()

            if i % self.print_interval == 0:
                logger.info(f'Train Step: {i}, Loss: {loss.item()}')

            # Calcolo dell'accuratezza
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(targets).sum().item()
            total_samples += targets.size(0)
            accuracy = total_correct / total_samples

            # Aggiungi la perdita e l'accuratezza a TensorBoard
            if self.writer is not None:
                self.writer.add_scalar('Loss/train', loss.item(), epoch * len(self.data_loader) + i)
                self.writer.add_scalar('Accuracy/train', accuracy, epoch * len(self.data_loader) + i)


    def evaluate(self, epoch=None, keep_all=False):
        self.model.eval()
        all_metrics = []
        total_correct = 0
        total_samples = 0

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
                total_correct += predicted.eq(targets).sum().item()
                total_samples += targets.size(0)
                accuracy = total_correct / total_samples

                # Logga la perdita e l'accuratezza di validazione su TensorBoard
                if self.writer is not None and epoch is not None and not self.is_training:
                    self.writer.add_scalar('Loss/val', loss.item(), epoch * len(self.data_loader) + i)
                    self.writer.add_scalar('Accuracy/val', accuracy, epoch * len(self.data_loader) + i)

        # Evito divisione per zero
        if len(all_metrics) == 0:
            return float('inf'), all_metrics  # Restituisco infinito se non ci sono metriche

        return sum(all_metrics) / len(all_metrics), all_metrics


def train_runner(config, model, trainer, val_evaluator, save_path):
    best_val_loss = float('inf')

    for epoch in range(config.epochs):  # Config non è dizionario, uso notazione a punto per accedere agli attributi
        logger.info(f'Epoch {epoch+1}/{config.epochs}')
        trainer.train_epoch(epoch)

        if epoch % config.val_interval == 0:
            val_loss, _ = val_evaluator.evaluate(epoch, keep_all = True)
            logger.info(f'Validation Loss: {val_loss}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                logger.info(f'Saved best model at epoch {epoch+1}')
