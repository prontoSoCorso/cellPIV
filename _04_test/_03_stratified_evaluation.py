#!/usr/bin/env python3
# stratified_evaluation_fixed.py
"""
Valutazione stratificata (adattamento del secondo script),
ma con i passi concreti e robusti del primo script per caricamento modelli
e predizioni (ROCKET, LSTMFCN, ConvTran).
"""

import os
import sys
import time
import copy
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
from tabulate import tabulate
from torch.utils.data import DataLoader, TensorDataset
import torch.multiprocessing as mp

warnings.filterwarnings("ignore", category=UserWarning)

# ==== project path resolution (come nei tuoi script) ====
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = current_dir
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

# ==== Imports from your project (come nel primo script) ====
from config import Config_03_train as conf_train
from config import Config_04_test as conf_test
from _03_train._b_LSTMFCN import TimeSeriesClassifier
from _03_train._c_ConvTranUtils import CustomDataset
from _99_ConvTranModel.model import model_factory
import _04_test._testFunctions as _testFunctions
import _utils_._utils as utils

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper: normalize column name for label (robusta)
def detect_label_col(df):
    # prefer uppercase 'BLASTO NY' then lowercase variations
    candidates = ["BLASTO NY", "blasto ny", "BLASTO_NY", "BLASTO", "blasto"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: try any column containing 'blasto' case-insensitive
    for c in df.columns:
        if 'blasto' in c.lower():
            return c
    raise ValueError("Colonna label contenente 'blasto' non trovata nel dataframe.")

# Helper: normalize PN values into strings like '1PN', '1.1PN', 'deg', etc.
def pn_to_str(x):
    if pd.isna(x):
        return x
    # if already string and endswith 'PN' or other category, keep as-is (strip spaces)
    if isinstance(x, str):
        s = x.strip()
        return s if s.endswith('PN') else s
    # numeric
    try:
        xf = float(x)
        if xf.is_integer():
            return f"{int(xf)}PN"
        else:
            # preserve decimal (e.g. 1.1 -> '1.1PN')
            s = str(xf).rstrip('0').rstrip('.')  # nicer formatting
            return f"{s}PN"
    except Exception:
        return str(x)

# ==== Model loading helpers (practical, based on working script) ====
def load_models_for_day(day, base_models_path, device):
    """
    Return dict of model_info keyed by model name.
    Each model_info is a dict containing at least:
      - 'type' : 'ROCKET'|'LSTMFCN'|'ConvTran'
      - model object (for PyTorch models) or classifier/transformer (for ROCKET)
      - 'threshold' : float
      - 'batch_size' if applicable
      - additional fields as needed
    """
    models = {}
    # ROCKET
    try:
        rocket_path = os.path.join(base_models_path, f"best_rocket_model_{day}Days.joblib")
        if os.path.exists(rocket_path):
            artifact = joblib.load(rocket_path)
            models['ROCKET'] = {
                'type': 'ROCKET',
                'classifier': artifact['classifier'],
                'transformer': artifact['rocket'],
                'threshold': artifact.get('final_threshold', 0.5)
            }
            print(f"Loaded ROCKET for {day} days from {rocket_path}")
    except Exception as e:
        print(f"Warning: failed to load ROCKET ({e})")

    # LSTMFCN
    try:
        lstm_path = os.path.join(base_models_path, f"best_lstmfcn_model_{day}Days.pth")
        if os.path.exists(lstm_path):
            # utils.load_lstmfcn_from_checkpoint returns (model, threshold, saved_params)
            lstm_model, lstm_threshold, saved_params = utils.load_lstmfcn_from_checkpoint(
                lstm_path, TimeSeriesClassifier, device=device
            )
            models['LSTMFCN'] = {
                'type': 'LSTMFCN',
                'model': lstm_model.to(device),
                'threshold': lstm_threshold if lstm_threshold is not None else 0.5,
                'batch_size': int(saved_params.get('batch_size', 128))
            }
            print(f"Loaded LSTMFCN for {day} days from {lstm_path}")
    except Exception as e:
        print(f"Warning: failed to load LSTMFCN ({e})")

    # ConvTran
    try:
        conv_path = os.path.join(base_models_path, f"best_convtran_model_{day}Days.pkl")
        if os.path.exists(conv_path):
            checkpoint = torch.load(conv_path, map_location=device, weights_only=False)
            # restore config
            conf_local = copy.deepcopy(conf_train)
            saved_config = checkpoint.get('config', {})
            for k, v in saved_config.items():
                setattr(conf_local, k, v)
            conv_model = model_factory(conf_local).to(device)
            conv_model.load_state_dict(checkpoint['model_state_dict'])
            conv_threshold = checkpoint.get('best_threshold', 0.5)
            models['ConvTran'] = {
                'type': 'ConvTran',
                'model': conv_model,
                'threshold': conv_threshold,
                'conf_local': conf_local
            }
            print(f"Loaded ConvTran for {day} days from {conv_path}")
    except Exception as e:
        print(f"Warning: failed to load ConvTran ({e})")

    return models

# ==== Prediction helper ====
def predict_with_model(model_info, X, y=None):
    """
    Generic prediction function for the model_info dict created above.
    X: numpy array of temporal features (N, ...). Could be (N, C, L) or (N, L).
    Returns y_pred (np.array int) and y_prob (np.array float).
    """
    mtype = model_info['type']
    if mtype == 'ROCKET':
        transformer = model_info['transformer']
        clf = model_info['classifier']
        threshold = model_info.get('threshold', 0.5)

        # --- Preparazione robusta di X per ROCKET ---
        def _prepare_X_for_rocket(X_in):
            X_proc = np.asarray(X_in)
            # Rimuovi dimensione inutili
            if X_proc.ndim == 1:
                X_proc = X_proc.reshape(1, -1)
            # Se è (N, L) aggiungi asse canali -> (N, 1, L)
            if X_proc.ndim == 2:
                X_proc = X_proc[:, None, :]
            if X_proc.ndim != 3:
                raise ValueError(f"Formato X non supportato per ROCKET: shape={X_proc.shape}; atteso (N, C, L) o (N, L)")
            # Sanitize NaN / Inf
            if not np.isfinite(X_proc).all():
                X_proc = np.nan_to_num(X_proc, nan=0.0, posinf=0.0, neginf=0.0)
            # Evita serie completamente costanti (std=0) che possono generare divisioni per zero in feature normalizzate
            # Aggiunge un piccolo rumore solo dove std==0.
            flat_std = X_proc.reshape(X_proc.shape[0], -1).std(axis=1)
            zero_std_idx = np.where(flat_std == 0)[0]
            if len(zero_std_idx) > 0:
                eps = 1e-6
                X_proc[zero_std_idx] += np.random.normal(0, eps, size=X_proc[zero_std_idx].shape)
            return X_proc

        X_prepared = _prepare_X_for_rocket(X)

        # sktime ROCKET si aspetta panel data: (n_instances, n_channels, n_timepoints)
        # Alcune implementazioni accettano (n_instances, n_timepoints) ma preferiamo forma esplicita.
        try:
            X_feat = transformer.transform(X_prepared)
        except ZeroDivisionError as zde:
            raise RuntimeError(f"ZeroDivisionError durante transformer.transform: possibile serie costante o kernel malformato. Shape X={X_prepared.shape}") from zde
        except SystemError as se:
            # Wrap con info diagnostica
            raise RuntimeError(f"SystemError interno Numba/ROCKET: shape={X_prepared.shape}; dtype={X_prepared.dtype}") from se

        probs = clf.predict_proba(X_feat)[:, 1]
        preds = (probs >= threshold).astype(int)
        return preds, probs

        
        artifact = joblib.load(model_path)
        model = artifact['classifier']
        transformer = artifact['rocket']
        threshold = artifact['final_threshold']
        
        X_features = transformer.transform(X)
        y_prob = model.predict_proba(X_features)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        return y_pred, y_prob

    elif mtype == 'LSTMFCN':
        model = model_info['model']
        threshold = model_info.get('threshold', 0.5)
        batch_size = model_info.get('batch_size', 128)

        # X must be torch tensor shaped as during training (N, C, L) or (N, L, 1) depending on your model.
        X_tensor = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, torch.tensor(np.zeros(len(X)), dtype=torch.long))  # labels not used here
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                            pin_memory=torch.cuda.is_available())

        model.eval()
        all_prob = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(device)
                output = model(X_batch)
                prob = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
                all_prob.extend(prob)
        probs = np.array(all_prob)
        preds = (probs >= threshold).astype(int)
        return preds, probs

    elif mtype == 'ConvTran':
        model = model_info['model']
        threshold = model_info.get('threshold', 0.5)
        conf_local = model_info.get('conf_local', conf_train)  # fallback

        # Ensure X has shape expected by CustomDataset
        # CustomDataset in your pipeline was called with (X, y) where X can be (N, C, L) or (N, L).
        dataset = CustomDataset(X, np.zeros(len(X)))
        loader = DataLoader(dataset, batch_size=getattr(conf_local, 'batch_size_convtran', 64), shuffle=False)
        model.eval()
        all_prob = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(device)
                output = model(X_batch)
                prob = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
                all_prob.extend(prob)
        probs = np.array(all_prob)
        preds = (probs >= threshold).astype(int)
        return preds, probs

    else:
        raise ValueError(f"Unknown model type: {mtype}")

# ==== Visual function (reused / adjusted from your second script) ====
# I keep visual_model_evaluation almost identical to your second script but
# it expects the df_perf csv to exist and df_merged passed in.
def visual_model_evaluation(csv_path, output_dir, merge_type, day, df_merged):
    import seaborn as sns
    import matplotlib.pyplot as plt
    df_perf = pd.read_csv(csv_path)

    print("Computing visual outputs (UMAP, bar plots, scatter)...")

    # fix label column detection
    label_col = detect_label_col(df_merged)
    df_merged = df_merged.copy()

    # Prepare UMAP per model (overall + strata)
    models = df_perf['Model'].unique()
    stratums = df_perf['Stratum'].unique()

    # lay out subplots
    n_cols = 3
    n_rows = int(np.ceil( (len(stratums)+1) / n_cols ))

    for model in models:
        plt.figure(figsize=(6*n_cols, 6*n_rows))
        plt.suptitle(f"UMAP - {model} ({day} Days, {merge_type} Merging)", y=1.02, fontsize=14)

        model_data = df_perf[(df_perf['Model'] == model) & (df_perf['Stratum'] == 'Overall')]
        if not model_data.empty:
            reducer = __import__("umap.umap_", fromlist=["UMAP"]).UMAP(random_state=42)
            temporal_cols = utils.detect_time_columns(df_merged)
            if len(temporal_cols) == 0:
                print("Warning: no temporal columns found for UMAP.")
                continue
            embedding = reducer.fit_transform(df_merged[temporal_cols].values)

            # Ground truth
            plt.subplot(n_rows, n_cols, 1)
            plt.scatter(embedding[:, 0], embedding[:, 1],
                        c=df_merged[label_col], cmap="coolwarm", alpha=0.7, s=15, vmin=0, vmax=1)
            plt.title(f"Ground Truth\n(n={len(df_merged)})", fontsize=10)
            plt.xticks([]); plt.yticks([])

            # Overall probs (expects a column f"{model}_prob" in df_merged)
            prob_col = f"{model}_prob"
            if prob_col in df_merged.columns:
                plt.subplot(n_rows, n_cols, 2)
                plt.scatter(embedding[:, 0], embedding[:, 1],
                            c=df_merged[prob_col], cmap="coolwarm", alpha=0.7, s=15, vmin=0, vmax=1)
                plt.title(f"Overall\n(n={len(df_merged)})", fontsize=10)
                plt.xticks([]); plt.yticks([])

            # Per stratum
            idx = 3
            for stratum in np.unique(df_merged['merged_PN']):
                mask = df_merged['merged_PN'] == stratum
                if mask.sum() == 0:
                    continue
                plt.subplot(n_rows, n_cols, idx)
                plt.scatter(embedding[mask, 0], embedding[mask, 1],
                            c=df_merged.loc[mask, prob_col] if prob_col in df_merged.columns else df_merged.loc[mask, label_col],
                            cmap="coolwarm", alpha=0.7, s=15, vmin=0, vmax=1)
                plt.title(f"{stratum}\n(n={mask.sum()})", fontsize=10)
                plt.xticks([]); plt.yticks([])
                idx += 1

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"umap_{model}_{day}Days_{merge_type}.png"), bbox_inches='tight', dpi=300)
        plt.close()

    # bar plots
    print("Computing bar plots...")
    def annotate_bars(ax):
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=8)

    metrics = ['balanced_accuracy', 'f1']
    import matplotlib.pyplot as plt
    import seaborn as sns
    for metric in metrics:
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x="Stratum", y=metric, hue="Model", data=df_perf)
        plt.title(f"{metric} by Embryo Type ({merge_type.capitalize()} Merging, {day} Days)", fontsize=14, pad=20)
        plt.ylabel(metric, fontsize=12); plt.xlabel("Embryo Type", fontsize=12)
        plt.ylim(0, 1.15)
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        annotate_bars(ax)
        filename = f"stratified_bar_plot_{metric.lower().replace(' ', '_')}_{day}Days_{merge_type}.png"
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
        plt.close()

    # scatter plot (balanced_accuracy vs f1)
    print("Computing scatter plot...")
    plt.figure(figsize=(14, 10))
    model_palette = {"ROCKET": "#1f77b4", "LSTMFCN": "#ff7f0e", "ConvTran": "#2ca02c"}
    scatter = sns.scatterplot(data=df_perf[df_perf["Stratum"] != "Overall"],
                              x="balanced_accuracy", y="f1", hue="Model", style="Stratum",
                              s=200, palette=model_palette, edgecolor="black", linewidth=0.8, alpha=0.9)
    for line in range(df_perf.shape[0]):
        if df_perf["Stratum"].iloc[line] != "Overall":
            plt.text(df_perf["balanced_accuracy"].iloc[line] + 0.015,
                     df_perf["f1"].iloc[line],
                     f"{df_perf['Stratum'].iloc[line]}", fontsize=10, ha='left', va='center',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
    plt.axhspan(0.7, 1.0, facecolor='#90EE90', alpha=0.25)
    plt.axhspan(0.4, 0.7, facecolor='#FFFF99', alpha=0.25)
    plt.axhspan(0.0, 0.4, facecolor='#FF9999', alpha=0.25)
    plt.title(f"Clinical Decision Matrix\n({merge_type.capitalize()} Merging, {day} Days)", fontsize=16, pad=20, weight='bold')
    plt.xlabel("balanced_accuracy", fontsize=14); plt.ylabel("f1", fontsize=14)
    plt.xlim(0.4, 1.05); plt.ylim(-0.05, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, framealpha=0.9, title="Model/Stratum",
               title_fontsize=12, fontsize=10, markerscale=1.5)
    filename = f"stratified_scatter_plot_{day}Days_{merge_type}.png"
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()

# ==== Main stratified evaluation function ====
def stratified_evaluation(days, model_types,
                          merge_type=("no_merging",), 
                          base_path=os.path.join(current_dir, "stratified_test_results", conf_train.method_optical_flow),
                          base_model_path=conf_test.base_model_path):

    os.makedirs(base_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # already set
        pass

    for day in days:
        print(f"\n=== Stratified evaluation for day {day} ===")
        # Load DB file path from config
        _, _, test_path = conf_test.get_paths(day)
        df_test = utils.load_data(test_path) if True else None
        if df_test is None or df_test.shape[0] == 0:
            print(f"No test data for day {day}. Skipping.")
            continue
        
        # In df test there is the column "PN"
        if "PN" not in df_test.columns:
            raise ValueError("La colonna 'PN' deve essere presente nel DataFrame di test.")
        
        # Use PN column to merge
        df_test = df_test.copy()
        df_test['PN_str'] = df_test['PN'].apply(pn_to_str)

        # Use PN and label to create summary table and check distribution
        df_summary = df_test.groupby(['PN_str', 'BLASTO NY']).size().unstack(fill_value=0)
        print("==================================================")
        print("TEST DISTRIBUTION AFTER MERGE")
        print(tabulate(df_summary, headers="keys", tablefmt="pretty"))
        print("==================================================")

        # start stratified analysis based on merge_type and per day
        output_path_per_day_and_merge = os.path.join(base_path, f"day {day} - {merge_type}")
        os.makedirs(output_path_per_day_and_merge, exist_ok=True)

        # Merge mapping. 1.1PN and 1PN to "1PN", 2.1PN and 2PN to "2PN", 3PN and >3PN to ">=3PN"
        merge_map = {
            "PN_number": {"1PN": ["1PN", "1.1PN"],
                          "2PN": ["2PN", "2.1PN"],
                          ">=3PN": ["3PN", ">3PN"]
                         },
            "anomalous": {"anomalous": ["1PN", "1.1PN", "2.1PN", "3PN", ">3PN"],
                          "normal": ["2PN"]
                         },
            "no_merging": None
        }.get(merge_type, None)

        # Apply merging
        df_merged = df_test.copy()
        
        # Debug: mostra valori PN unici nel dataset per verificare il mapping
        unique_pn_values = sorted(df_merged['PN_str'].dropna().unique())
        print(f"Valori PN unici nel dataset: {unique_pn_values}")
        
        if merge_map:
            # Debug: verifica che tutti i valori PN siano mappati
            all_mapped_values = set()
            for k, v in merge_map.items():
                all_mapped_values.update(v)
            unmapped_values = set(unique_pn_values) - all_mapped_values
            if unmapped_values:
                print(f"ATTENZIONE: Valori PN non mappati: {unmapped_values}")
            
            # create merged_PN column
            df_merged['merged_PN'] = df_merged['PN_str'].apply(lambda x: next((k for k, v in merge_map.items() if x in v), x))
        else:
            # no merging, keep original PN_str
            df_merged['merged_PN'] = df_merged['PN_str']

        # show distribution after merge
        label_col = detect_label_col(df_test)
        table2 = df_merged.groupby("merged_PN")[label_col].value_counts().unstack(fill_value=0)
        print("==================================================")
        print("TEST STRATIFICATION AFTER MERGE")
        print(tabulate(table2, headers="keys", tablefmt="pretty"))
        print("==================================================")
        print(f"Numero di istanze test prima del merge: {df_test.shape[0]}")
        print(f"Numero di istanze dopo il merge: {df_merged.shape[0]}")
        print("==================================================")
        table2_path = os.path.join(output_path_per_day_and_merge, "test_distribution_after_merge.txt")
        with open(table2_path, 'w') as f:
            f.write("TEST STRATIFICATION AFTER MERGE\n")
            f.write(tabulate(table2, headers="keys", tablefmt="pretty"))

        # Prepare X, y (temporal cols detection)
        data_dict = utils.build_data_dict(df_test=df_merged)
        _, _, _, _, X_all, y_all = utils._check_data_dict(data_dict, 
                                                          require_test=True, 
                                                          only_check_test=True)
        
        # Sanitize input
        X_all = utils._sanitize_np(X_all)
        # Ground truth as 1D numpy array
        y_all = np.asarray(y_all).astype(int).ravel()

        # Detect temporal columns
        temporal_cols = utils.detect_time_columns(df_merged)
        if len(temporal_cols) == 0:
            raise ValueError("Nessuna colonna temporale 'value_' trovata nel DataFrame.")

        # Update conf for ConvTran shape/labels (value_for_config_convTran esiste)
        conv_config = _testFunctions.value_for_config_convTran(X_all)
        conf_train.num_labels = conv_config["num_labels"]
        conf_train.Data_shape = conv_config["data_shape"]

        # Load models (practical loader)
        models = load_models_for_day(day=day, base_models_path=base_model_path, device=device)
        # Filter by requested model_types
        models = {k: v for k, v in models.items() if k in model_types}

        # Evaluate each model: overall + per stratum
        results = []
        for model_name, model_info in models.items():
            print(f"\nEvaluating model {model_name} for day {day}...")

            # Overall
            try:
                y_pred_overall, y_prob_overall = predict_with_model(model_info, X_all, y_all)
            except Exception as e:
                print(f"Error predicting overall for {model_name}: {e}")
                continue

            metrics_overall = utils.calculate_metrics(y_all, y_pred_overall, y_prob_overall)
            df_merged[f"{model_name}_prob"] = y_prob_overall  # store probs for visualizations
            results.append({
                "Model": model_name,
                "Stratum": "Overall",
                "balanced_accuracy": metrics_overall.get('balanced_accuracy', np.nan),
                "f1": metrics_overall.get('f1', np.nan)
            })

            # Per-group evaluation
            roc_data = []
            cm_data = []
            for group, group_df in df_merged.groupby("merged_PN"):
                n_group = group_df.shape[0]
                if n_group < 2:
                    print(f"Skipping group {group} (n={n_group}) for {model_name} — troppo piccolo.")
                    continue
                X_grp = group_df[temporal_cols].values
                X_grp = utils._sanitize_np(X_grp)
                y_grp = group_df[label_col].values.astype(int)

                try:
                    y_pred_grp, y_prob_grp = predict_with_model(model_info, X_grp, y_grp)
                except Exception as e:
                    print(f"Prediction failed for group {group}, model {model_name}: {e}")
                    continue

                metrics_grp = utils.calculate_metrics(y_grp, y_pred_grp, y_prob_grp)
                results.append({
                    "Model": model_name,
                    "Stratum": group,
                    "balanced_accuracy": metrics_grp.get('balanced_accuracy', np.nan),
                    "f1": metrics_grp.get('f1', np.nan)
                })

                # collect for summary plots if present
                roc_data.append((group, metrics_grp.get('fpr', []), metrics_grp.get('tpr', []), metrics_grp.get('roc_auc', np.nan)))
                cm_data.append((group, metrics_grp.get('conf_matrix', np.array([[0,0],[0,0]]))))

            # summary plots per model
            if roc_data:
                try:
                    _testFunctions.plot_summary_roc_curves(model_name, roc_data, day, output_path_per_day_and_merge)
                except Exception as e:
                    print(f"Warning: plot_summary_roc_curves failed for {model_name}: {e}")
            if cm_data:
                try:
                    _testFunctions.plot_summary_confusion_matrices(model_name, cm_data, day, output_path_per_day_and_merge)
                except Exception as e:
                    print(f"Warning: plot_summary_confusion_matrices failed for {model_name}: {e}")

        # Save CSV results
        result_file = os.path.join(output_path_per_day_and_merge, f"stratified_model_performance_{merge_type}_{day}Days.csv")
        pd.DataFrame(results).to_csv(result_file, index=False)
        print(f"\n📁 Results saved to {result_file}")

        # Visual results (UMAP, bar plots, scatter)
        try:
            visual_model_evaluation(csv_path=result_file, output_dir=output_path_per_day_and_merge, merge_type=merge_type, day=day, df_merged=df_merged)
        except Exception as e:
            print(f"Warning: visual_model_evaluation failed: {e}")

    print("All done.")


# ==== If invoked as script ====
if __name__ == "__main__":
    start_time = time.time()
    # default run: no merging, days 1,3,5, all 3 models
    merge_type = "PN_number"   # oppure "PN_number"
    days_to_consider = [1, 3, 5]
    model_types = ["ROCKET", "LSTMFCN", "ConvTran"]

    stratified_evaluation(days=days_to_consider, model_types=model_types, merge_type=merge_type)
    print(f"Execution time: {time.time() - start_time:.2f} seconds")
