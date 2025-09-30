import os
import sys
import pandas as pd
import numpy as np
import torch
import copy
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader
import umap.umap_ as umap

# Aggiungo il percorso del progetto al sys.path
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = current_dir
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

# Import config and model definitions
from config import Config_03_train as conf
from _03_train._b_LSTMFCN import TimeSeriesClassifier
from _03_train._c_ConvTranUtils import CustomDataset
import _04_test._testFunctions as _testFunctions
from _99_ConvTranModel.model import model_factory
import _utils_._utils as utils

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _sanitize_np(X):
    X = np.asarray(X, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

def test_all(base_path = os.path.join(current_dir, "plots_and_metrics_test"),
             days=[1, 3, 5],
             models = ['ROCKET', 'LSTMFCN', 'ConvTran'],
             base_models_path=conf.output_model_base_dir,
             method_optical_flow=conf.method_optical_flow):
    
    # Create base output directory
    base_path_opt_flow = os.path.join(base_path, method_optical_flow)
    os.makedirs(base_path_opt_flow, exist_ok=True)

    # Instance arrays
    roc_data = []
    metrics_data = []
    umap_results = []

    # For each day, load test data, models, make predictions, compute metrics
    for day in days:
        print(f"\nProcessing day {day}...")
        output_path_per_day = os.path.join(base_path_opt_flow,f"day {day}")
        os.makedirs(output_path_per_day, exist_ok=True)
        
        # Load test data
        # Prepara i dati (lettura e organizzazione esterna)
        _, _, test_path = conf.get_paths(day)
        df_test = utils.load_data(test_path) if True else None  # se vuoi disattivare il test, metti False
        data_dict = utils.build_data_dict(df_test=df_test)
        _, _, _, _, X_test, y_test = utils._check_data_dict(data=data_dict, 
                                                           require_test=True,
                                                           only_check_test=True)
        # Sanitize input
        X = _sanitize_np(X_test)        
        # Ground truth as 1D numpy array
        y_true = np.asarray(y_test).astype(int).ravel()

        # Store data for UMAP and Confusion matrices
        day_umap_data = {
            'X': X,
            'y_true': y_true,
            'models': {}
        }
        conf_matrices = {}

        for model_name in models:
            # Load model and predict
            if model_name == 'ROCKET':
                model_path = os.path.join(base_models_path, f"best_rocket_model_{day}Days.joblib")
                artifact = joblib.load(model_path)
                model = artifact['classifier']
                transformer = artifact['rocket']
                threshold = artifact['final_threshold']
                
                X_features = transformer.transform(X)
                y_prob = model.predict_proba(X_features)[:, 1]
                y_pred = (y_prob >= threshold).astype(int)

            elif model_name == 'LSTMFCN':
                model_path = os.path.join(base_models_path, f"best_lstmfcn_model_{day}Days.pth")

                # Rebuild model and load best threshold
                model, threshold, saved_params = utils.load_lstmfcn_from_checkpoint(model_path, TimeSeriesClassifier, device=device)
                model = model.to(device)
                batch_size = int(saved_params.get('batch_size', 128))

                # Prepare DataLoader
                X_tensor = torch.tensor(X, dtype=torch.float32)
                dataset = TensorDataset(X_tensor, torch.tensor(y_true))
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                    num_workers=0, pin_memory=torch.cuda.is_available())
                
                # Predict
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

                # Restore config from checkpoint
                conf_local = copy.deepcopy(conf)
                saved_config = checkpoint['config']
                for key, value in saved_config.items():
                    setattr(conf_local, key, value)

                # Rebuild model and load state dict
                model = model_factory(conf_local).to(device)
                model.load_state_dict(checkpoint['model_state_dict'])
                threshold = checkpoint.get('best_threshold', 0.5)

                # Prepare DataLoader
                dataset = CustomDataset(X, y_true)
                loader = DataLoader(dataset, batch_size=conf_local.batch_size_convtran, shuffle=False)
                
                # Predict
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

            else:
                raise ValueError(f"Modello non supportato: {model_name}")

            # Compute metrics
            metrics = utils.calculate_metrics(y_true, y_pred, y_prob)
            # Store confusion matrix
            conf_matrices[model_name] = metrics['conf_matrix']
            # Store predictions for UMAP
            day_umap_data['models'][model_name] = {
                'prob': y_prob,
                'pred': y_pred,  # Store threshold-based predictions
                'threshold': threshold  # Store the actual threshold used
                }

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


        umap_results.append((day, day_umap_data))

        # ---------------------------
        # Save combined confusion matrices
        # ---------------------------
        print(f"\n📊 Creating combined confusion matrix for day {day}")
        fig, axs = plt.subplots(1, len(models), figsize=(6*len(models), 6))
        if len(models) == 1:  # Handle single model case
            axs = [axs]
        
        
        for idx, (model_name, cm) in enumerate(conf_matrices.items()):
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues', ax=axs[idx],
                annot_kws={"size": 16}  # Increase font size for numbers
                )
            axs[idx].set_xticklabels(["no_blasto", "blasto"], fontsize=16)
            axs[idx].set_yticklabels(["no_blasto", "blasto"], fontsize=16)
            axs[idx].set_title(f'{model_name}\n{day} Days')
            axs[idx].set_xlabel('Predicted', fontsize=16)
            axs[idx].set_ylabel('Actual', fontsize=16)
            axs[idx].tick_params(axis='both', which='major', labelsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(output_path_per_day, f'combined_confusion_matrix_{day}Days.png'), 
                   bbox_inches='tight')
        plt.close()


    ##############################
    # Plot ROC curves
    ##############################
    print("\nGenerating ROC curves...")
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
    plt.legend(loc='lower right', frameon=True)
    plt.savefig(os.path.join(base_path_opt_flow, 'roc_curves.png'), bbox_inches='tight')
    plt.close()

    ##############################
    # Generate bar plots for each day
    ##############################
    print("\nGenerating bar plots...")
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
        plt.ylim((0,1))
        plt.xticks(index + bar_width, metrics)
        plt.legend()
        output_path_per_day = os.path.join(base_path_opt_flow,f"day {day}")
        plt.savefig(os.path.join(output_path_per_day, f'summary_metrics_{day}Days.png'))
        plt.close()


    ##############################
    # UMAP visualization
    ##############################
    print("\nGenerating UMAP visualizations...")

    # Create figure
    n_days = len(days)
    n_cols = len(models) + 1  # "+1" because of the ground truth
    fig = plt.figure(figsize=(6*n_cols + 2, 6*n_days))  # +2 for colorbar space
    gs = fig.add_gridspec(n_days, n_cols + 1, width_ratios=[1]*n_cols + [0.1])
    
    # Create consistent colorbar
    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)

    for day_idx, (day, data) in enumerate(umap_results):
        # Prepare X for UMAP: flatten + PCA + UMAP (robust default)
        X_arr = data['X']  # (N, C, L) or (N, L)
        y_true_day = data['y_true']
        if X_arr.ndim == 3:
            N, C, L = X_arr.shape
            X_flat = X_arr.reshape(N, C * L)
        elif X_arr.ndim == 2:
            X_flat = X_arr
        else:
            raise ValueError(f"Unexpected X shape {X_arr.shape}")

        # Standardize
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_flat)

        # PCA to 50 dims (or fewer if features < 50)
        pca_n = min(50, X_scaled.shape[1])
        pca = PCA(n_components=pca_n, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        # UMAP on PCA features
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding = reducer.fit_transform(X_pca)  # shape (N, 2)

        # Plot ground truth in first column
        ax = fig.add_subplot(gs[day_idx, 0])
        sc = ax.scatter(embedding[:, 0], embedding[:, 1], 
                       c=y_true_day, cmap="coolwarm", 
                       alpha=0.7, s=15, edgecolors='face',
                       vmin=0, vmax=1)
        ax.set_title(f"Ground Truth - {day} Days", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add ground truth contours for reference
        for label in [0, 1]:
            mask = (y_true_day == label)
            if mask.sum() > 10:
                sns.kdeplot(x=embedding[mask, 0], y=embedding[mask, 1],
                           levels=3, color='blue' if label == 0 else 'red',
                           alpha=0.5, ax=ax)
        
        # Plot model predictions in subsequent columns
        for model_idx, model_name in enumerate(models, start=1):
            ax = fig.add_subplot(gs[day_idx, model_idx])
            model_data = data['models'][model_name]
            
            # Plot UMAP colored by predictions
            ax.scatter(
                embedding[:, 0], embedding[:, 1], 
                c=model_data['prob'], cmap="coolwarm", 
                alpha=0.7, s=15, edgecolors='face',
                vmin=0, vmax=1
                )

            ax.set_title(f"{model_name} - {day} Days\n(Threshold: {model_data['threshold']:.2f})", 
                         fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add ground truth contours
            for label in [0, 1]:
                mask_pred = (model_data['pred'] == label)
                if mask_pred.sum() > 10:  # Only plot contours if enough points
                    sns.kdeplot(x=embedding[mask_pred, 0], y=embedding[mask_pred, 1],
                               levels=3, color='blue' if label == 0 else 'red',
                               alpha=0.5, ax=ax)

    # Add colorbar in new column
    cbar_ax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Blasto Probability\n(0=Negative, 1=Positive)", 
                 rotation=270, labelpad=25, fontsize=12)

    plt.subplots_adjust(right=0.85)  # Adjust for colorbar space
    plt.savefig(os.path.join(base_path_opt_flow, 'umap_summary.png'), 
               bbox_inches='tight', dpi=300)
    plt.close()
    print("📊 Saved UMAP summary visualization")


if __name__ == "__main__":
    test_all()
