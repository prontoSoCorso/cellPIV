import os
import sys
import argparse
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
from _03_train._c_ConvTranUtils import CustomDataset
import _04_test._myFunctions as _myFunctions

def predict(model_type: str, model: Any, X: np.ndarray, params: Dict, device: torch.device) -> np.ndarray:
    """Generate predictions for different model types"""
    if model_type == "ROCKET":
        X_3d = X[:, np.newaxis, :]
        X_features = params['transformer'].transform(X_3d)
        y_prob = model.predict_proba(X_features)[:, 1]
        return (y_prob >= params['threshold']).astype(int)

    # PyTorch model prediction
    if model_type == "LSTMFCN":
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    else:
        dataset = CustomDataset(X.reshape(X.shape[0], 1, -1), np.zeros(len(X)))  # Dummy labels

    loader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=False)
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
                 model_names: Tuple[str, str], output_dir: str) -> float:
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
    
    # Save visualization
    img_path = os.path.join(output_dir, f"mcnemar_{model_names[0]}_vs_{model_names[1]}.png")
    _myFunctions.save_contingency_matrix_with_mcnemar(
        contingency_table, img_path, model_names[0], model_names[1], result.pvalue
    )

    # Print results
    print(f"\nMcNemar's Test Results ({' vs '.join(model_names)}):")
    print(f"Contingency Table:\n{np.array(contingency_table)}")
    print(f"p-value: {result.pvalue:.4f}")
    print("Statistically significant" if result.pvalue < 0.05 else "No significant difference")
    
    return result.pvalue

def main(model_type: str, days: Tuple[int, int], data_dir: str, model_dir: str, output_dir: str):
    """Main comparison workflow"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # Load both models
    models = []
    for day in days:
        test_path = os.path.join(data_dir, f"Normalized_sum_mean_mag_{day}Days_test.csv")
        X, y = _myFunctions.prepare_data(model_type=model_type, test_path=test_path)
        model_info = _myFunctions.load_model_by_type(model_type=model_type, days=day, base_models_path=model_dir, device=device, data=X)
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
    return mcnemar_test(models[0][3], *predictions, model_names, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform McNemar's test between two models")
    parser.add_argument("--model-type", type=str, required=True, choices=["ROCKET", "LSTMFCN", "ConvTran"])
    parser.add_argument("--days", type=int, nargs=2, required=True, help="Pair of days to compare")
    parser.add_argument("--data-dir", type=str, default=parent_dir)
    parser.add_argument("--model-dir", type=str, default=current_dir)
    parser.add_argument("--output-dir", type=str, default=os.path.join(current_dir, "mcnemar_results"))
    check = True

    try:
        args = parser.parse_args()
    except:
        check = False
        model_type = "LSTMFCN"
        days = [1,3]
        data_dir = parent_dir
        model_dir = current_dir
        output_dir = os.path.join(current_dir, "mcnemar_results")
    
    main(
        model_type= args.model_type if check else model_type,
        days=args.days if check else days,
        data_dir=args.data_dir if check else data_dir,
        model_dir=args.model_dir if check else model_dir,
        output_dir=os.path.join(args.output_dir if check else output_dir, args.model_type if check else model_type)
        )