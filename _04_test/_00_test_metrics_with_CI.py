import os
import sys
import pandas as pd
import numpy as np
import torch
import copy
import joblib
import math
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
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
             method_optical_flow=conf_train.method_optical_flow,
             n_bootstrap=1000,
             sample_frac=1.0,
             random_seed=42):
    
    # Create base output directory
    base_path_opt_flow = os.path.join(base_path, method_optical_flow)
    os.makedirs(base_path_opt_flow, exist_ok=True)

    # Instance arrays
    roc_data = []
    metrics_data = []
    umap_results = []
    
    # Store bootstrap summaries for final CSV
    bootstrap_summary_all = []

    # For each day, load test data, models, make predictions, compute metrics
    for day in days:
        print(f"\nProcessing day {day}...")
        output_path_per_day = os.path.join(base_path_opt_flow,f"day {day}")
        os.makedirs(output_path_per_day, exist_ok=True)
        
        # Load test data
        _, _, test_path = conf_train.get_paths(day)
        df_test = utils.load_data(test_path) if True else None
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
            print(f"  > Evaluating {model_name}...")
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
                model, threshold, saved_params = utils.load_lstmfcn_from_checkpoint(model_path, TimeSeriesClassifier, device=device)
                model = model.to(device)
                batch_size = int(saved_params.get('batch_size', 128))

                X_tensor = torch.tensor(X, dtype=torch.float32)
                dataset = TensorDataset(X_tensor, torch.tensor(y_true))
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                    num_workers=0, pin_memory=torch.cuda.is_available())
                
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

                conf_local = copy.deepcopy(conf_train)
                saved_config = checkpoint['config']
                for key, value in saved_config.items():
                    setattr(conf_local, key, value)

                model = model_factory(conf_local).to(device)
                model.load_state_dict(checkpoint['model_state_dict'])
                threshold = checkpoint.get('best_threshold', 0.5)

                dataset = CustomDataset(X, y_true)
                loader = DataLoader(dataset, batch_size=conf_local.batch_size_convtran, shuffle=False)
                
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

            # ---------------------------
            # Standard Metrics
            # ---------------------------
            metrics = utils.calculate_metrics(y_true, y_pred, y_prob)
            conf_matrices[model_name] = metrics['conf_matrix']
            
            day_umap_data['models'][model_name] = {
                'prob': y_prob,
                'pred': y_pred,
                'threshold': threshold
            }

            roc_data.append({
                'model': model_name,
                'day': day,
                'fpr': metrics['fpr'],
                'tpr': metrics['tpr'],
                'auc': metrics['roc_auc']
            })

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
            # BOOTSTRAP IMPLEMENTATION
            # ---------------------------
            
            # Setup bootstrap variables
            n_samples = len(y_true)
            sample_size = max(1, int(math.ceil(sample_frac * n_samples)))
            rng = np.random.RandomState(random_seed)
            
            # Metrics to track in bootstrap
            target_metrics = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'brier']
            boot_arr = {k: [] for k in target_metrics}

            # Run Bootstrap loop
            for _ in tqdm(range(n_bootstrap), desc=f"    Bootstrap {model_name}", leave=False):
                # Sample indices with replacement
                idxs = rng.randint(0, n_samples, size=sample_size)
                
                yt_boot = y_true[idxs]
                yp_prob_boot = y_prob[idxs]
                # Re-apply threshold to get predictions for this sample set
                yp_pred_boot = (yp_prob_boot >= threshold).astype(int)
                
                # Check if we have at least one sample of each class to avoid errors in AUC
                if len(np.unique(yt_boot)) < 2:
                    # Skip iteration or fill nan (skipping is usually safer for AUC stability)
                    continue

                # Calculate metrics for this fold
                # Note: We assume utils.calculate_metrics returns a dict with these keys
                m_boot = utils.calculate_metrics(yt_boot, yp_pred_boot, yp_prob_boot)
                
                for k in target_metrics:
                    val = m_boot.get(k, float('nan'))
                    boot_arr[k].append(val)

            # Compute Confidence Intervals (95%)
            ci_results = {}
            for k, arr in boot_arr.items():
                arr_valid = np.array([x for x in arr if not np.isnan(x)])
                if arr_valid.size > 0:
                    ci_lower = float(np.percentile(arr_valid, 2.5))
                    ci_upper = float(np.percentile(arr_valid, 97.5))
                    std_dev = float(np.std(arr_valid))
                else:
                    ci_lower, ci_upper, std_dev = np.nan, np.nan, np.nan
                
                ci_results[k] = {
                    'lower': ci_lower,
                    'upper': ci_upper,
                    'std': std_dev,
                    'values': arr_valid # Keep for plotting
                }

                # Add to summary list for CSV
                bootstrap_summary_all.append({
                    'Day': day,
                    'Model': model_name,
                    'Metric': k,
                    'Value': metrics.get(k, np.nan), # Original value on full test set
                    'CI_Lower_95': ci_lower,
                    'CI_Upper_95': ci_upper,
                    'Std_Dev': std_dev
                })

            # Plot Bootstrap Distributions
            n_plots = len(target_metrics)
            cols = 3
            rows = math.ceil(n_plots / cols)
            
            fig_boot, axs_boot = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
            axs_boot = axs_boot.flatten()
            
            for i, metric_name in enumerate(target_metrics):
                data = ci_results[metric_name]['values']
                observed_val = metrics.get(metric_name, np.nan)
                
                if len(data) > 0:
                    sns.histplot(data, kde=True, ax=axs_boot[i], color='skyblue', edgecolor='black')
                    axs_boot[i].axvline(observed_val, color="red", linestyle="--", linewidth=2, label=f"Observed: {observed_val:.3f}")
                    axs_boot[i].axvline(ci_results[metric_name]['lower'], color="green", linestyle=":", label="95% CI")
                    axs_boot[i].axvline(ci_results[metric_name]['upper'], color="green", linestyle=":")
                    axs_boot[i].set_title(f"{metric_name}\nCI: [{ci_results[metric_name]['lower']:.3f}, {ci_results[metric_name]['upper']:.3f}]")
                    axs_boot[i].legend(fontsize='small')
                else:
                    axs_boot[i].set_title(f"{metric_name} (No Data)")
            
            # Remove empty subplots
            for j in range(i+1, len(axs_boot)):
                fig_boot.delaxes(axs_boot[j])
                
            plt.tight_layout()
            plt.savefig(os.path.join(output_path_per_day, f'bootstrap_dist_{model_name}_{day}Days.png'))
            plt.close()


        # ---------------------------
        # Save combined confusion matrices (Existing Code)
        # ---------------------------
        print(f"\n📊 Creating combined confusion matrix for day {day}")
        fig, axs = plt.subplots(1, len(models), figsize=(6*len(models), 6))
        if len(models) == 1: axs = [axs]
        
        for idx, (model_name, cm) in enumerate(conf_matrices.items()):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[idx], annot_kws={"size": 16})
            axs[idx].set_xticklabels(["no_blasto", "blasto"], fontsize=16)
            axs[idx].set_yticklabels(["no_blasto", "blasto"], fontsize=16)
            axs[idx].set_title(f'{model_name}\n{day} Days')
            axs[idx].set_xlabel('Predicted', fontsize=16)
            axs[idx].set_ylabel('Actual', fontsize=16)

        plt.tight_layout()
        plt.savefig(os.path.join(output_path_per_day, f'combined_confusion_matrix_{day}Days.png'), bbox_inches='tight')
        plt.close()

        umap_results.append((day, day_umap_data))

    # ---------------------------
    # Save Bootstrap Summary to CSV
    # ---------------------------
    print(f"\nSaving Bootstrap Summary CSV...")
    df_boot_summary = pd.DataFrame(bootstrap_summary_all)
    csv_path = os.path.join(base_path_opt_flow, "bootstrap_metrics_summary.csv")
    df_boot_summary.to_csv(csv_path, index=False)
    print(f"Summary saved to: {csv_path}")

    # ---------------------------
    # Plot ROC curves (Existing Code)
    # ---------------------------
    print("\nGenerating ROC curves...")
    plt.figure(figsize=(10, 8))
    colors = {'ConvTran': 'blue', 'ROCKET': 'green', 'LSTMFCN': 'red'}
    linestyles = {1: '-', 3: '--', 5: '-.', 7: ':'}
    
    for entry in roc_data:
        model = entry['model']
        day = entry['day']
        plt.plot(entry['fpr'], entry['tpr'], color=colors[model], linestyle=linestyles[day],
                 label=f'{model} {day} Days (AUC={entry["auc"]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc='lower right', frameon=True)
    plt.savefig(os.path.join(base_path_opt_flow, 'roc_curves.png'), bbox_inches='tight')
    plt.close()

    # ---------------------------
    # Generate bar plots (Existing Code)
    # ---------------------------
    print("\nGenerating bar plots...")
    df_metrics = pd.DataFrame(metrics_data)
    for day in days:
        day_df = df_metrics[df_metrics['day'] == day]
        plt.figure(figsize=(12, 6))
        bar_width = 0.25
        metrics_list = ['accuracy', 'balanced_accuracy', 'auc', 'precision', 'recall', 'f1']
        index = np.arange(len(metrics_list))
        
        for i, model in enumerate(day_df['model'].unique()):
            model_data = day_df[day_df['model'] == model]
            values = [model_data[metric].values[0] for metric in metrics_list]
            bars = plt.bar(index + i * bar_width, values, bar_width, label=model)
            
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}', 
                        ha='center', va='bottom', fontsize=10)
        
        plt.title(f'Metrics Comparison for {day} Days')
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.ylim((0,1))
        plt.xticks(index + bar_width, metrics_list)
        plt.legend()
        plt.savefig(os.path.join(base_path_opt_flow, f"day {day}", f'summary_metrics_{day}Days.png'))
        plt.close()

    # ---------------------------
    # FIXED: Metrics Summary Overview
    # ---------------------------
    # Create a list to store rows (replacing the deprecated .append method)
    summary_rows = []
    
    for day in days:
        for model in models:
            df_subset = df_boot_summary[(df_boot_summary['Day'] == day) & (df_boot_summary['Model'] == model)]
            
            if df_subset.empty:
                continue

            # Check if rows exist before accessing
            auc_rows = df_subset[df_subset['Metric'] == 'roc_auc']
            f1_rows = df_subset[df_subset['Metric'] == 'f1']
            
            if not auc_rows.empty and not f1_rows.empty:
                auc_row = auc_rows.iloc[0]
                f1_row = f1_rows.iloc[0]
                
                # Append dictionary to list
                summary_rows.append({
                    "classifier": model,
                    "hours": day * 24,
                    "auc_mean": auc_row['Value'],
                    "auc_ci_low": auc_row['CI_Lower_95'],
                    "auc_ci_high": auc_row['CI_Upper_95'],
                    "f1_mean": f1_row['Value'],
                    "f1_ci_low": f1_row['CI_Lower_95'],
                    "f1_ci_high": f1_row['CI_Upper_95']
                })
    
    # Create DataFrame once at the end
    df_summary = pd.DataFrame(summary_rows, columns=["classifier", "hours", "auc_mean", "auc_ci_low", "auc_ci_high", "f1_mean", "f1_ci_low", "f1_ci_high"])
    
    summary_csv_path = os.path.join(base_path_opt_flow, "metrics_summary_overview.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"Metrics summary overview saved to: {summary_csv_path}")


if __name__ == "__main__":
    test_all()