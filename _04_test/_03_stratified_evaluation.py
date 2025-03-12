import os
import sys
import warnings
import pandas as pd
import torch
import numpy as np
import joblib
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.multiprocessing as mp

warnings.filterwarnings("ignore", category=UserWarning)

# Path configuration
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = current_dir
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_03_train as conf
from _03_train._c_ConvTranUtils import CustomDataset
from _03_train._b_LSTMFCN import LSTMFCN
from _99_ConvTranModel.model import model_factory
import _04_test._myFunctions as _myFunctions


def load_models(day, model_path, device):
    """Load all models for a specific day"""
    models = {}
    
    try:
        # Load ROCKET
        rocket_path = os.path.join(model_path, f"best_rocket_model_{day}Days.joblib")
        rocket = joblib.load(rocket_path)
        models['ROCKET'] = {
            'model': rocket['classifier'],
            'transformer': rocket['rocket'],
            'threshold': rocket['final_threshold']
            }

        # Load LSTMFCN
        lstm_path = os.path.join(model_path, f"best_lstmfcn_model_{day}Days.pth")
        lstm_checkpoint = torch.load(lstm_path, map_location=device, weights_only=False)
        models['LSTMFCN'] = {
            'model': LSTMFCN(**{k: lstm_checkpoint[k] for k in ['lstm_size', 'filter_sizes',
                                                                'kernel_sizes', 'dropout', 'num_layers']}
                            ).to(device),
            'threshold': lstm_checkpoint.get('best_threshold', 0.5),
            'batch_size': lstm_checkpoint['batch_size']
            }
        models['LSTMFCN']['model'].load_state_dict(lstm_checkpoint['model_state_dict'])

        # Load ConvTran
        conv_path = os.path.join(model_path, f"best_convtran_model_{day}Days.pkl")
        conv_checkpoint = torch.load(conv_path, map_location=device, weights_only=False)
        models['ConvTran'] = {
            'model': model_factory(conf).to(device),
            'threshold': conv_checkpoint.get('best_threshold', 0.5)
            }
        models['ConvTran']['model'].load_state_dict(conv_checkpoint['model_state_dict'])

    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise

    return models



def evaluate_model(model_type, model_info, X, y, temporal_cols, device):
    """Generic model evaluation function"""
    if model_type == 'ROCKET':
        y_pred, y_prob = _myFunctions.test_model_ROCKET(model_info=model_info, X=X)
        return y_pred, y_prob
    
    # PyTorch model evaluation
    if model_type == 'LSTMFCN':
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32).unsqueeze(-1), torch.tensor(y))
        loader = DataLoader(dataset, batch_size=model_info['batch_size'], shuffle=False)
    else:  # ConvTran
        dataset = CustomDataset(X.reshape(X.shape[0], 1, -1), y)
        loader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=False)
    
    model_info['model'].eval()
    all_pred, all_prob = [], []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            output = model_info['model'](X_batch)
            prob = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
            all_pred.extend((prob >= model_info['threshold']).astype(int))
            all_prob.extend(prob)
    
    return np.array(all_pred), np.array(all_prob)


def save_distribution_table(table, filename, title=""):
    """Save pandas table to a text file with tabulate formatting"""
    table_str = f"{title}\n{tabulate(table, headers='keys', tablefmt='grid')}\n\n"
    with open(filename, 'w') as f:
        f.write(table_str)


def visual_model_evaluation(csv_path, output_dir, merge_type, day):
    import seaborn as sns
    import matplotlib.pyplot as plt

    df_perf = pd.read_csv(csv_path)
    """Create clinical-grade visualizations with detailed annotations"""
    
    # Helper function for annotation
    def annotate_bars(ax):
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}", 
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', 
                       xytext=(0, 5), 
                       textcoords='offset points',
                       fontsize=8)

    # 1. Create bar plots for each metric
    metrics = ['balanced_accuracy', 'f1']
    for metric in metrics:
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x="Stratum", y=metric, hue="Model", data=df_perf)
        plt.title(f"{metric} by Embryo Type ({merge_type.capitalize()} Merging, {day} Days)", 
                 fontsize=14, pad=20)
        plt.ylabel(metric, fontsize=12)
        plt.xlabel("Embryo Type", fontsize=12)
        plt.ylim(0, 1.15)
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        annotate_bars(ax)
        
        # Save with dynamic naming
        filename = f"stratified_bar_plot_{metric.lower().replace(' ', '_')}_{day}Days_{merge_type}.png"
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
        plt.close()

    # 2. Create detailed scatter plot
    plt.figure(figsize=(14, 10))

    # Use a more distinct color palette
    model_palette = {
        "ROCKET": "#1f77b4",  # Distinct blue
        "LSTMFCN": "#ff7f0e",  # Bright orange
        "ConvTran": "#2ca02c"  # Strong green
    }

    # Create scatter plot with enhanced visibility
    scatter = sns.scatterplot(
        data=df_perf[df_perf["Stratum"] != "Overall"],
        x="balanced_accuracy", y="f1",
        hue="Model", style="Stratum",
        s=200, palette=model_palette,
        edgecolor="black", linewidth=0.8,  # Thicker borders
        alpha=0.9  # Slightly more opaque
    )

    # Add detailed annotations with better contrast
    for line in range(df_perf.shape[0]):
        if df_perf["Stratum"].iloc[line] != "Overall":
            plt.text(df_perf["balanced_accuracy"].iloc[line] + 0.015,  # Slightly more offset
                    df_perf["f1"].iloc[line],
                    f"{df_perf['Stratum'].iloc[line]}",
                    fontsize=10, ha='left', va='center',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

    # Enhanced clinical decision zones
    plt.axhspan(0.7, 1.0, facecolor='#90EE90', alpha=0.25)  # Safe zone
    plt.axhspan(0.4, 0.7, facecolor='#FFFF99', alpha=0.25)  # Caution zone
    plt.axhspan(0.0, 0.4, facecolor='#FF9999', alpha=0.25)  # High-risk zone

    # Add zone labels
    plt.text(0.45, 0.85, "Safe Zone", fontsize=12, color='darkgreen', alpha=0.8)
    plt.text(0.45, 0.55, "Caution Zone", fontsize=12, color='darkgoldenrod', alpha=0.8)
    plt.text(0.45, 0.25, "High-Risk Zone", fontsize=12, color='darkred', alpha=0.8)

    # Plot formatting
    plt.title(f"Clinical Decision Matrix\n({merge_type.capitalize()} Merging, {day} Days)", 
            fontsize=16, pad=20, weight='bold')
    plt.xlabel("balanced_accuracy", fontsize=14)
    plt.ylabel("f1", fontsize=14)
    plt.xlim(0.4, 1.05)
    plt.ylim(-0.05, 1.05)

    # Enhanced legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
            frameon=True, framealpha=0.9,
            title="Model/Stratum", title_fontsize=12,
            fontsize=10, markerscale=1.5)

    # Save scatter plot
    filename = f"stratified_scatter_plot_{day}Days_{merge_type}.png"
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()


def stratified_evaluation(merge_types, days=1, 
         base_path=os.path.join(current_dir, "final_stratified_test_results"), 
         base_model_path=current_dir,
         base_test_csv_path=parent_dir,
         db_file=os.path.join(parent_dir, "DB morpheus UniPV.xlsx")
         ):
    os.makedirs(base_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mp.set_start_method("spawn", force=True)  # Imposta multiprocessing con spawn

    # ---------------------------
    # Caricamento del DB completo
    # ---------------------------
    df_db = pd.read_excel(db_file)
    # Verifica che le colonne di interesse siano presenti (es. "slide_well" e "PN")
    if "slide_well" not in df_db.columns or "PN" not in df_db.columns:
        raise ValueError("Le colonne 'slide_well' e 'PN' devono essere presenti nel file DB morpheus UniPV.xlsx.")
    
    table1 = df_db.groupby("PN")["blasto ny"].value_counts().unstack()
    print("==================================================")
    print("FULL DB DISTRIBUTION")
    print(tabulate(table1, headers="keys", tablefmt="pretty"))
    table1_path = os.path.join(base_path, "full_db_distribution.txt")
    save_distribution_table(table1, table1_path, "FULL DB DISTRIBUTION")
    
    '''
    Le diverse tipologie di pronuclei sono:
    - PN = 0 (Apronucleato, fecondazione fallita)
    - PN = 1 (Monopronucleato, L'embrione ha un solo pronucleo invece dei due attesi--> anomalo, mancato sviluppo del pronucleo maschile o femminile)
    - PN = 1.1 (Indica una variante di embrioni non facilmente e direttamente associabili al gruppo 1PN e 2PN, una via di mezzo)
    - PN = 2 (Dipropronucleato, normale)
    - PN = 2.1 (Indica una variante di embrioni non facilmente e direttamente associabili al gruppo 2PN e 3PN, una via di mezzo)
    - PN = 3 (Tripronucleato, probabile polispermia o problemi nella estrusione del secondo globulo polare. Generalmente non trasferiti perché scarsissima possibilità sviluppo sano)
    - PN = >3 (Polipronucleato, anomalo, disfunzionale)
    - PN = deg (embrioni non vitali, con severi danni cellulari, no divisione cellulare o eccessiva frammentazione)
    '''
    
    import itertools
    for merge_type, day in itertools.product(merge_types, days):
        output_path_per_day_and_merge = os.path.join(base_path,f"day {day} - {merge_type}")
        os.makedirs(output_path_per_day_and_merge, exist_ok=True)
        
        # ---------------------------
        # Caricamento del test set da stratificare
        # ---------------------------
        df_test = pd.read_csv(os.path.join(base_test_csv_path, f"Normalized_sum_mean_mag_{day}Days_test.csv"))

        # ---------------------------
        # Merge categories and merging test
        # ---------------------------
        merge_map = {
            "anomalous": {"1PN", "1.1PN", "2.1PN", "3PN", ">3PN"},  # tutti i fecondati anomali
            "not_vital": {"0PN", "deg"},                            # Unisco i non vitali e lascio separato il resto
            "no_merging": None                                      # Faccio analisi stratificata per tutti
            }.get(merge_type, None)
        
        if merge_map:
            df_db["merged_PN"] = df_db["PN"].apply(
                lambda x: merge_type if x in merge_map else x)
        else:
            df_db["merged_PN"] = df_db["PN"]

        # Uniamo il DB e il test usando:
        #   DB: "slide_well" (che contiene il riferimento al well)
        #   Test: "dish_well"
        df_merged = pd.merge(
            df_test, 
            df_db[["slide_well", "merged_PN"]], 
            left_on="dish_well", 
            right_on="slide_well"
        ).dropna(subset=["merged_PN"])
        
        table2 = df_merged.groupby("merged_PN")[("BLASTO NY").upper()].value_counts().unstack()
        print("==================================================")
        print("TEST STRATIFICATION AFTER MERGE")
        print(tabulate(table2, headers="keys", tablefmt="pretty"))
        print("==================================================")
        print(f"Numero di istanze test prima del merge: {df_test.shape[0]}")
        print(f"Numero di istanze dopo il merge: {df_merged.shape[0]}")
        print("==================================================")
        table2_path = os.path.join(output_path_per_day_and_merge, "test_distribution_after_merge.txt")
        save_distribution_table(table2, table2_path, "TEST STRATIFICATION AFTER MERGE")
        
        # ---------------------------
        # Preparing data and test
        # ---------------------------
        # Prepare data
        temporal_cols = [c for c in df_merged.columns if c.startswith("value_")]
        X = df_merged[temporal_cols].values
        y = df_merged["BLASTO NY"].values

        # Aggiorno conf.Data_shape ed il num_labels in base ad X
        conf.num_labels = _myFunctions.value_for_config_convTran(X)["num_labels"]
        conf.Data_shape = _myFunctions.value_for_config_convTran(X)["data_shape"]

        # Load models
        models = load_models(day=day, model_path=base_model_path, device=device)

        # Evaluate models
        results = []
        for model_name, model_info in models.items():
            # Overall evaluation
            y_pred, y_prob = evaluate_model(model_name, model_info, X, y, temporal_cols, device)
            metrics = _myFunctions.calculate_metrics(y, y_pred, y_prob)
            results.append({
                "Model": model_name,
                "Stratum": "Overall",
                **{k: v for k, v in metrics.items() if k in ["balanced_accuracy", "f1"]}
                })

            # Stratified evaluation
            # For summarized ROC and CM plots, accumulate subgroup data
            roc_data = []
            cm_data = []
            for group, group_df in df_merged.groupby("merged_PN"):
                X_grp = group_df[temporal_cols].values
                y_grp = group_df["BLASTO NY"].values
                y_pred_grp, y_prob_grp = evaluate_model(model_name, model_info, X_grp, y_grp, temporal_cols, device)
                metrics_grp = _myFunctions.calculate_metrics(y_grp, y_pred_grp, y_prob_grp)
                results.append({
                    "Model": model_name,
                    "Stratum": group,
                    **{k: v for k, v in metrics_grp.items() if k in ["balanced_accuracy", "f1"]}
                    })
                
                # Collect ROC and CM data for summary plots
                roc_data.append((group, metrics_grp["fpr"], metrics_grp["tpr"], metrics_grp["roc_auc"]))
                cm_data.append((group, metrics_grp["conf_matrix"]))
                
            # Create a single summary ROC plot for this model & day
            if roc_data:
                _myFunctions.plot_summary_roc_curves(model_name, roc_data, day, output_path_per_day_and_merge)
            # Create a single summary confusion matrix plot for this model & day
            if cm_data:
                _myFunctions.plot_summary_confusion_matrices(model_name, cm_data, day, output_path_per_day_and_merge)


        # Save results
        result_file = os.path.join(output_path_per_day_and_merge, f"stratified_model_performance_{merge_type}_{day}Days.csv")
        pd.DataFrame(results).to_csv(result_file, index=False)
        print(f"\n📁 Results saved to {result_file}")

        # ---------------------------
        # Visual Results
        # ---------------------------
        print("\nComputing visual results...\n")
        visual_model_evaluation(csv_path=result_file, output_dir=output_path_per_day_and_merge, merge_type=merge_type, day=day)


if __name__ == "__main__":
    import time
    start_time = time.time()

    merge_types = ["anomalous", "not_vital"]    # "anomalous" OR "not_vital" OR "no_merging"
    days_to_consider = [1,3,5,7]        # 1,3,5,7
    
    stratified_evaluation(merge_types=merge_types, days=days_to_consider)
    print(f"Execution time: {time.time() - start_time:.2f} seconds\n")
