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
from config import Config_03_train as conf_train
from _03_train._b_LSTMFCN import TimeSeriesClassifier
from _03_train._c_ConvTranUtils import CustomDataset
import _04_test._testFunctions as _testFunctions
from _99_ConvTranModel.model import model_factory
import _utils_._utils as utils

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_all(base_path = os.path.join(current_dir, "plots_and_metrics_test"),
             days=[1, 3, 5],
             models = ['ROCKET', 'LSTMFCN', 'ConvTran'],
             base_models_path=conf_train.output_model_base_dir,
             method_optical_flow=conf_train.method_optical_flow):
    
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
        _, _, test_path = conf_train.get_paths(day)
        df_test = utils.load_data(test_path) if True else None  # se vuoi disattivare il test, metti False
        data_dict = utils.build_data_dict(df_test=df_test)
        _, _, _, _, X_test, y_test = utils._check_data_dict(data=data_dict, 
                                                           require_test=True,
                                                           only_check_test=True)
        # Sanitize input
        X = utils._sanitize_np(X_test)        
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
                conf_local = copy.deepcopy(conf_train)
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
    # UMAP visualization - TRUE BLUE & ORANGE
    ##############################
    print("\nGenerating UMAP visualizations...")
    import matplotlib.colors as mcolors

    # --- ACCESSIBILITY SETTINGS ---
    # Define the specific colors
    color_class_0 = '#000080' # Navy Blue (Dark) -> Represents 0 (Negative)
    color_class_1 = '#FF8000' # Vivid Orange (Bright) -> Represents 1 (Positive)

    # CREATE CUSTOM COLORMAP: Transitions from Navy Blue to Vivid Orange
    # This replaces 'cividis' so the points match the contours exactly.
    cmap_blue_orange = mcolors.LinearSegmentedColormap.from_list(
        "BlueOrange", [color_class_0, "#DDDDDD", color_class_1] 
    )
    # Note: I added a light gray middle point ("#DDDDDD") to make the 
    # transition smoother and the middle probability (0.5) distinct, 
    # but you can remove it if you want a direct gradient.
    
    # --------------------------------------

    # Create figure
    n_days = len(days)
    n_cols = len(models) + 1 
    fig = plt.figure(figsize=(6*n_cols + 2, 6*n_days)) 
    gs = fig.add_gridspec(n_days, n_cols + 1, width_ratios=[1]*n_cols + [0.1])
    
    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap=cmap_blue_orange, norm=norm)

    for day_idx, (day, data) in enumerate(umap_results):
        X_arr = data['X']  
        y_true_day = data['y_true']
        if X_arr.ndim == 3:
            N, C, L = X_arr.shape
            X_flat = X_arr.reshape(N, C * L)
        elif X_arr.ndim == 2:
            X_flat = X_arr
        else:
            raise ValueError(f"Unexpected X shape {X_arr.shape}")

        # Standardize & PCA & UMAP
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_flat)
        pca_n = min(50, X_scaled.shape[1])
        pca = PCA(n_components=pca_n, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding = reducer.fit_transform(X_pca) 

        # --- PLOT GROUND TRUTH ---
        ax = fig.add_subplot(gs[day_idx, 0])
        ax.scatter(embedding[:, 0], embedding[:, 1], 
                   c=y_true_day, cmap=cmap_blue_orange, 
                   alpha=0.8, s=25, edgecolors='none', vmin=0, vmax=1)
        
        ax.set_title(f"Ground Truth - Day {day}", fontsize=24)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Blue/Orange Contours
        for label, color_code in zip([0, 1], [color_class_0, color_class_1]):
            mask = (y_true_day == label)
            if mask.sum() > 10:
                sns.kdeplot(x=embedding[mask, 0], y=embedding[mask, 1],
                           levels=3, color=color_code, linewidths=2,
                           alpha=0.6, ax=ax)
        
        # --- PLOT MODELS ---
        for model_idx, model_name in enumerate(models, start=1):
            ax = fig.add_subplot(gs[day_idx, model_idx])
            model_data = data['models'][model_name]
            
            ax.scatter(embedding[:, 0], embedding[:, 1], 
                c=model_data['prob'], cmap=cmap_blue_orange, 
                alpha=0.7, s=25, edgecolors='none', vmin=0, vmax=1)
            
            ax.set_title(f"{model_name} - Day {day}", fontsize=24, pad=8)
            ax.set_xticks([])
            ax.set_yticks([])
            
            for label, color_code in zip([0, 1], [color_class_0, color_class_1]):
                mask_pred = (model_data['pred'] == label)
                if mask_pred.sum() > 10:  
                    sns.kdeplot(x=embedding[mask_pred, 0], y=embedding[mask_pred, 1],
                               levels=3, color=color_code, linewidths=2,
                               alpha=0.6, ax=ax)

    # --- COLORBAR ---
    cbar_ax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    
    cbar.set_label("Blasto Probability\n(0=No-Blasto, 1=Blasto)", 
                 rotation=270, labelpad=60, fontsize=24)
    
    cbar.ax.tick_params(labelsize=18)

    plt.subplots_adjust(right=0.85)
    plt.savefig(os.path.join(base_path_opt_flow, 'umap_summary.png'), 
               bbox_inches='tight', dpi=500)
    plt.close()
    print("📊 Saved TRUE BLUE-ORANGE UMAP visualization")



if __name__ == "__main__":
    import time
    start_time = time.time()
    test_all()
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

    # around 25 seconds
