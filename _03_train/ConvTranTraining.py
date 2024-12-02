import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, cohen_kappa_score, brier_score_loss, confusion_matrix


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
            accuracy = total_correct / total_samples

            # Aggiungi la perdita e l'accuratezza a TensorBoard
            if self.writer is not None:
                self.writer.add_scalar('Loss/train', loss.item(), epoch * len(self.data_loader) + i)
                self.writer.add_scalar('Accuracy/train', accuracy, epoch * len(self.data_loader) + i)

            # Stampa la perdita e l'accuratezza nel terminale
            if i % self.print_interval == 0:
                print(f'Train Step: {i}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

    def evaluate(self, epoch=None, keep_all=False):
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
    def evaluate_test(self, save_conf_matrix=False, conf_matrix_filename='conf_matrix_ConvTran.png'):
        self.model.eval()
        targets_list = []
        preds_list = []
        probs_list = []

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                # Calcolo delle predizioni
                _, predicted = outputs.max(1)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[:, 1]  # Probabilità per la classe positiva

                # Salva predizioni e target
                preds_list.extend(predicted.cpu().numpy())
                targets_list.extend(targets.cpu().numpy())
                probs_list.extend(probs)

        # Calcolo metriche
        accuracy = accuracy_score(targets_list, preds_list)
        balanced_accuracy = balanced_accuracy_score(targets_list, preds_list)
        kappa = cohen_kappa_score(targets_list, preds_list)
        brier = brier_score_loss(targets_list, probs_list, pos_label=1)
        f1 = f1_score(targets_list, preds_list, average='binary')
        cm = confusion_matrix(targets_list, preds_list)

        # Stampa metriche
        print(f"=====ConvTran Test Metrics=====\n"
              f"Accuracy: {accuracy}\n"
              f"Balanced Accuracy: {balanced_accuracy}\n"
              f"Cohen Kappa: {kappa}\n"
              f"Brier Score: {brier}\n"
              f"F1 Score: {f1}\n")

        if save_conf_matrix:
            self.save_confusion_matrix(cm, conf_matrix_filename)

        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'kappa': kappa,
            'brier': brier,
            'f1_score': f1,
            'confusion_matrix': cm
        }

    # Metodo per salvare la matrice di confusione
    @staticmethod
    def save_confusion_matrix(cm, filename):
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False, xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig(filename)
        plt.close()


def train_runner(config, model, trainer, val_evaluator, save_path):
    best_val_loss = float('inf')

    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizer, mode='min', factor=config.scheduler_factor, patience=config.scheduler_patience)

    for epoch in range(config.epochs):  # Config non è dizionario, uso notazione a punto per accedere agli attributi
        print(f'Epoch {epoch+1}/{config.epochs}')
        trainer.train_epoch(epoch)

        if epoch % config.val_interval == 0:
            val_loss, _ = val_evaluator.evaluate(epoch, keep_all = True)
            print(f'Validation Loss: {val_loss}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                print(f'Saved best model at epoch {epoch+1}')