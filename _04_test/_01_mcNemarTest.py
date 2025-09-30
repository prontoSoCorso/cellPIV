import os
import sys
import matplotlib.pyplot as plt
import torch
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, Any

# Configure project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = current_dir
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

# Model-specific imports
from config import Config_03_train as conf
from _03_train._c_ConvTranUtils import CustomDataset
import _04_test._testFunctions as _testFunctions

def predict(model_type: str, model: Any, X: np.ndarray, params: Dict, device: torch.device) -> np.ndarray:
    """Generate predictions for different model types"""
    if model_type.upper() == "ROCKET":
        X_3d = X[:, np.newaxis, :]
        X_features = params['transformer'].transform(X_3d)
        y_prob = model.predict_proba(X_features)[:, 1]
        return (y_prob >= params['threshold']).astype(int)

    # PyTorch model prediction
    if model_type.upper() == "LSTMFCN":
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=False)
    else:
        dataset = CustomDataset(X.reshape(X.shape[0], 1, -1), np.zeros(len(X)))  # Dummy labels
        loader = DataLoader(dataset, batch_size=conf.batch_size_convtran, shuffle=False)

    model.eval()
    preds = []
    
    with torch.no_grad():
        for X_batch in loader:
            X_batch = X_batch[0].to(device)
            output = model(X_batch)
            prob = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
            preds.extend((prob >= params['threshold']).astype(int))
    
    return np.array(preds)

def mcnemar_test(y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray, 
                 model_names: Tuple[str, str]) -> float:
    """Perform McNemar's test and save results"""
    # Create contingency table
    correct_1 = (pred1 == y_true)
    correct_2 = (pred2 == y_true)
    
    contingency_table = [
        [np.sum(~correct_1 & ~correct_2), np.sum(~correct_1 & correct_2)],
        [np.sum(correct_1 & ~correct_2), np.sum(correct_1 & correct_2)]
    ]

    # Perform statistical test
    result = mcnemar(contingency_table, exact=True)
    
    # Print results
    print(f"\nMcNemar's Test Results ({' vs '.join(model_names)}):")
    print(f"Contingency Table:\n{np.array(contingency_table)}")
    print(f"p-value: {result.pvalue:.4f}")
    print("Statistically significant" if result.pvalue < 0.05 else "No significant difference")
    
    return result.pvalue, np.array(contingency_table), model_names

def create_combined_contingency_plot(contingency_data: list, output_dir: str):
    """Create a combined plot with multiple contingency matrices"""
    n = len(contingency_data)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6), constrained_layout=True)

    for idx, (contingency_table, model_names, p_value) in enumerate(contingency_data):
        ax = axes[idx] if n > 1 else axes
        im = ax.imshow(contingency_table, cmap='Blues')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(contingency_table[i, j]), 
                        ha='center', va='center',
                        fontsize=14, color='white' if contingency_table[i, j] > np.max(contingency_table) / 2 else 'black')
                        
        # Configure axes
        ax.set_title(f"{model_names[0]} vs {model_names[1]}\np-value: {p_value:.2e}")
        ax.set_xlabel(f'{model_names[1]} Predictions', fontsize=12)
        ax.set_ylabel(f'{model_names[0]} Predictions')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Model 2 Wrong", "Model 2 Correct"], fontsize=12)
        ax.set_yticklabels(["Model 1 Wrong", "Model 1 Correct"], fontsize=12)

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        cbar.ax.tick_params(labelsize=12)

    plt.savefig(os.path.join(output_dir, "combined_contingency_matrices.png"), dpi=300)
    plt.close()

def compare_with_McNemar(models_type: str, days: Tuple[int, int], data_dir: str, model_dir: str, output_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    contingency_data = []
    
    for model_type in models_type:
        """Main comparison workflow"""
        # Load both models
        models = []
        for day in days:
            df_test = _testFunctions.load_test_data(day)
            X, y = _testFunctions.prepare_data(model_type=model_type, df=df_test)
            model_info = _testFunctions.load_model_by_type(model_type=model_type, days=day, base_models_path=model_dir, device=device, data=(X,y))
            model = model_info["model"]
            params = {key: value for key, value in model_info.items() if key != "model"}
            models.append((model, params, X, y))

        # Verify consistent test labels
        if not np.array_equal(models[0][3], models[1][3]):
            raise ValueError("Test labels differ between days - cannot perform comparison")

        # Generate predictions
        predictions = [
            predict(model_type, model, X, params, device)
            for model, params, X, _ in models
            ]

        # Perform statistical test
        model_names = [f"{model_type}_{day}Days" for day in days]
        p_value, contingency_table, model_names = mcnemar_test(models[0][3], *predictions, model_names)
        contingency_data.append((contingency_table, model_names, p_value))

    # Create combined plot
    create_combined_contingency_plot(contingency_data, output_dir)

if __name__ == "__main__":
    models_type = ["ROCKET","LSTMFCN","ConvTran"]
    days = [1,3]
    data_dir = parent_dir
    model_dir = current_dir
    base_output_dir = os.path.join(current_dir, "mcnemar_results")
    
    compare_with_McNemar(
        models_type=models_type,
        days=days,
        data_dir=data_dir,
        model_dir=model_dir,
        output_dir=base_output_dir
        )