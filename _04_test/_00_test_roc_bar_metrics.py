import os
import sys
import pandas as pd
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_curve, auc

# Aggiungo il percorso del progetto al sys.path
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = current_dir
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

# Import config and model definitions
from config import Config_03_train as conf
from _03_train._b_LSTMFCN import LSTMFCN
from _03_train._c_ConvTranUtils import CustomDataset
from _99_ConvTranModel.model import model_factory
import _04_test._myFunctions as _myFunctions

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_all(base_path = os.path.join(current_dir, "final_test_images_and_metrics"), 
         days=[1, 3, 5, 7], models = ['ROCKET', 'LSTMFCN', 'ConvTran'], base_models_path=current_dir, base_test_csv_path=parent_dir):
    os.makedirs(base_path, exist_ok=True)

    roc_data = []
    metrics_data = []

    for day in days:
        output_path_per_day = os.path.join(base_path,f"day {day}")
        os.makedirs(output_path_per_day, exist_ok=True)

        # Load test data
        test_csv = os.path.join(base_test_csv_path, f"Normalized_sum_mean_mag_{day}Days_test.csv")
        df_test = pd.read_csv(test_csv)
        temporal_columns = [col for col in df_test.columns if col.startswith('value_')]
        X = df_test[temporal_columns].values
        y_true = df_test['BLASTO NY'].values

        for model_name in models:
            # Load model and predict
            if model_name == 'ROCKET':
                model_path = os.path.join(base_models_path, f"best_rocket_model_{day}Days.joblib")
                artifact = joblib.load(model_path)
                model = artifact['classifier']
                transformer = artifact['rocket']
                threshold = artifact['final_threshold']
                
                X_3d = X[:, np.newaxis, :]
                X_features = transformer.transform(X_3d)
                y_prob = model.predict_proba(X_features)[:, 1]
                y_pred = (y_prob >= threshold).astype(int)

            elif model_name == 'LSTMFCN':
                model_path = os.path.join(base_models_path, f"best_lstmfcn_model_{day}Days.pth")
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)

                model = LSTMFCN(
                    lstm_size=checkpoint['lstm_size'],
                    filter_sizes=checkpoint['filter_sizes'],
                    kernel_sizes=checkpoint['kernel_sizes'],
                    dropout=checkpoint['dropout'],
                    num_layers=checkpoint['num_layers']
                ).to(device)
                model.load_state_dict(checkpoint['model_state_dict'])
                threshold = checkpoint.get('best_threshold', 0.5)
                batch_size = checkpoint['batch_size']

                X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
                dataset = TensorDataset(X_tensor, torch.tensor(y_true))
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                
                model.eval()
                all_pred, all_prob = [], []
                with torch.no_grad():
                    for X_batch, _ in loader:
                        X_batch = X_batch.to(device)
                        output = model(X_batch)
                        prob = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
                        pred = (prob >= threshold).astype(int)
                        all_pred.extend(pred)
                        all_prob.extend(prob)
                y_pred, y_prob = np.array(all_pred), np.array(all_prob)

            elif model_name == 'ConvTran':
                model_path = os.path.join(base_models_path, f"best_convtran_model_{day}Days.pkl")
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                conf.num_labels = 2
                conf.Data_shape = (1, X.shape[1])
                model = model_factory(conf).to(device)
                model.load_state_dict(checkpoint['model_state_dict'])
                threshold = checkpoint.get('best_threshold', 0.5)
                
                X_conv = X.reshape(X.shape[0], 1, X.shape[1])
                dataset = CustomDataset(X_conv, y_true)
                loader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=False)
                
                model.eval()
                all_pred, all_prob = [], []
                with torch.no_grad():
                    for X_batch, _ in loader:
                        X_batch = X_batch.to(device)
                        output = model(X_batch)
                        prob = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
                        pred = (prob >= threshold).astype(int)
                        all_pred.extend(pred)
                        all_prob.extend(prob)
                y_pred, y_prob = np.array(all_pred), np.array(all_prob)

            # Compute metrics
            metrics = _myFunctions.calculate_metrics(y_true, y_pred, y_prob)
            
            # ---------------------------
            # Save confusion matrix
            # ---------------------------
            plt.figure(figsize=(6,6))
            sns.heatmap(metrics['conf_matrix'], annot=True, fmt='d', cmap='Blues',  
                        xticklabels=["no_blasto", "blasto"], yticklabels=["no_blasto", "blasto"])
            plt.title(f'Confusion Matrix - {model_name} ({day} Days)')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.savefig(os.path.join(output_path_per_day, f'confusion_matrix_{model_name}_{day}Days.png'))
            plt.close()

            # Collect ROC data
            roc_data.append({
                'model': model_name,
                'day': day,
                'fpr': metrics['fpr'],
                'tpr': metrics['tpr'],
                'auc': metrics['roc_auc']
            })

            # Collect metrics
            metrics_data.append({
                'model': model_name,
                'day': day,
                'accuracy': metrics['accuracy'],
                'balanced_accuracy': metrics['balanced_accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'brier': metrics['brier'],
                'auc': metrics['roc_auc']
            })


    # ---------------------------
    # Plot ROC curves
    # ---------------------------
    plt.figure(figsize=(10, 8))
    colors = {'ConvTran': 'blue', 'ROCKET': 'green', 'LSTMFCN': 'red'}
    linestyles = {1: '-', 3: '--', 5: '-.', 7: ':'}
    
    for entry in roc_data:
        model = entry['model']
        day = entry['day']
        plt.plot(entry['fpr'], entry['tpr'],
                 color=colors[model],
                 linestyle=linestyles[day],
                 label=f'{model} {day} Days (AUC={entry["auc"]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(base_path, 'roc_curves.png'), bbox_inches='tight')
    plt.close()

    # ---------------------------
    # Generate bar plots for each day
    # ---------------------------
    df_metrics = pd.DataFrame(metrics_data)
    for day in days:
        day_df = df_metrics[df_metrics['day'] == day]
        plt.figure(figsize=(12, 6))
        bar_width = 0.25
        metrics = ['accuracy', 'balanced_accuracy', 'auc', 'precision', 'recall', 'f1']
        index = np.arange(len(metrics))
        
        for i, model in enumerate(day_df['model'].unique()):
            model_data = day_df[day_df['model'] == model]
            values = [model_data[metric].values[0] for metric in metrics]
            bars = plt.bar(index + i * bar_width, values, bar_width, label=model)
            
            # Annotate bars with their values
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}', 
                        ha='center', va='bottom', fontsize=10)
        
        plt.title(f'Metrics Comparison for {day} Days')
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.xticks(index + bar_width, metrics)
        plt.legend()
        output_path_per_day = os.path.join(base_path,f"day {day}")
        plt.savefig(os.path.join(output_path_per_day, f'summary_metrics_{day}Days.png'))
        plt.close()


if __name__ == "__main__":
    test_all(base_path = os.path.join(current_dir, "final_test_images_and_metrics"), 
         days=[1, 3, 5, 7], models = ['ROCKET', 'LSTMFCN', 'ConvTran'], base_models_path=current_dir, base_test_csv_path=parent_dir)




