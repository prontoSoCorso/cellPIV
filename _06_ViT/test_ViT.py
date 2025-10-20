import torch
import h5py
import numpy as np
from model_ViT import VisionTransformer
from config_ViT import *
from train_ViT import EmbryoDataset
from torch.utils.data import DataLoader
import math
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, accuracy_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score, brier_score_loss, f1_score, confusion_matrix

def calculate_metrics(y_true, y_pred, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc,
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "brier": brier_score_loss(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
        "conf_matrix": confusion_matrix(y_true, y_pred),
        "fpr": fpr,
        "tpr": tpr
    }

    decimals = 4
    for key, value in metrics.items():
        if isinstance(value, float):
            metrics[key] = round(value, decimals)
        elif isinstance(value, np.ndarray):
            metrics[key] = np.round(value, decimals)

    return metrics

def plot_summary_confusion_matrices(model_name, cm_data, day, output_dir):
    n = len(cm_data)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten() if n > 1 else [axes]
    for i, (group, cm) in enumerate(cm_data):
        ax = axes[i]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_title(f"{group}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(f"{model_name} Confusion Matrices ({day} Days)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_file = os.path.join(output_dir, f"{model_name}_conf_matrix_{day}Days.png")
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()

def plot_summary_roc_curves(model_name, roc_data, day, output_dir):
    plt.figure(figsize=(8, 6))
    for group, fpr, tpr, roc_auc in roc_data:
        plt.plot(fpr, tpr, lw=2, label=f"{group} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"{model_name} ROC Curve ({day} Days)", fontsize=14)
    plt.legend(loc="lower right")
    output_file = os.path.join(output_dir, f"{model_name}_ROC_{day}Days.png")
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()

def test():
    test_dataset = EmbryoDataset(DATA_PATH, 'test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = VisionTransformer().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "best_model.pth")))
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1])
            all_labels.extend(labels.numpy())

    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    
    # Plot confusion matrix
    cm_data = [("Test Set", metrics["conf_matrix"])]
    plot_summary_confusion_matrices("ViT", cm_data, "24.0", RESULTS_DIR)
    
    # Plot ROC curve
    roc_data = [("Test Set", metrics["fpr"], metrics["tpr"], metrics["roc_auc"])]
    plot_summary_roc_curves("ViT", roc_data, "24.0", RESULTS_DIR)
    
    # Save metrics
    with open(os.path.join(RESULTS_DIR, "test_metrics.txt"), "w") as f:
        for key, value in metrics.items():
            if key not in ['conf_matrix', 'fpr', 'tpr']:
                f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    test()