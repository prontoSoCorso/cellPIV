#!/usr/bin/env python3
'''
test_with_GradCAM_fixed2.py

Usage: drop in your cellPIV repo and run:
    python _04_test/test_with_GradCAM_fixed2.py
'''

import os
import sys
import copy
import json
import re
import csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# minimize TF chatter (optional)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# --- project root discovery (same as in your repo) ---
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = current_dir
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

# --- imports from your project ---
from config import Config_03_train as conf_train
from _03_train._b_LSTMFCN import TimeSeriesClassifier
from _99_ConvTranModel.model import model_factory
import _utils_._utils as utils
from _modelAdapter import ModelAdapter
from signal_grad_cam import TorchCamBuilder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------- User config ----------------
DAYS = [3]                         # days to process
MODELS_TO_RUN = ['ConvTran']  # choose subset
USE_SMALL_SUBSETS = False              # for quick tests with smaller data
OUTPUT_BASE = os.path.join(current_dir, "GRADCAM_batch_outputs_stratified")
os.makedirs(OUTPUT_BASE, exist_ok=True)

EXPLAINER_TYPES = ["Grad-CAM"]     # or ["Grad-CAM","HiResCAM"]
TARGET_LAYERS_OVERRIDE = {'LSTMFCN': None, 'ConvTran': None}  # set to layer name if needed
SAVE_PER_SAMPLE_VIS = False        # If True -> files are kept; if False -> use tempdir and delete
ACQUISITION_STEP_HOURS = 0.25     # 15 minutes = 0.25 hours
# ------------------------------------------------

# ---------- Helpers ----------
def model_has_conv2d(model, layer_name=None):
    '''Return True if model (or the specified layer module) contains Conv2d.'''
    import torch.nn as nn
    if layer_name is not None:
        # find module by name
        for n, m in model.named_modules():
            if n == layer_name:
                return isinstance(m, nn.Conv2d) or any(isinstance(sub, nn.Conv2d) for sub in m.modules())
        return False
    else:
        for _, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                return True
    return False


def model_has_conv1d(model, layer_name=None):
    import torch.nn as nn
    if layer_name is not None:
        for n, m in model.named_modules():
            if n == layer_name:
                return isinstance(m, nn.Conv1d) or any(isinstance(sub, nn.Conv1d) for sub in m.modules())
        return False
    else:
        for _, m in model.named_modules():
            if isinstance(m, nn.Conv1d):
                return True
    return False


def normalize_sample_for_model(sample, use_conv2d=False):
    '''
    Convert arbitrary-sample numpy array -> appropriate shape depending on model:
      - if use_conv2d True: return shape (C, H=1, W=T)  (so batch -> (N,C,1,T) valid 4D for Conv2d)
      - otherwise return shape (C, T)
    '''
    a = np.asarray(sample)
    if a.ndim == 0:
        base = np.array([[float(a)]])           # (1,1)
    elif a.ndim == 1:
        base = a[np.newaxis, :]                # (1, T)
    else:
        # assume time is the longest axis
        time_axis = int(np.argmax(a.shape))
        if time_axis != a.ndim - 1:
            a = np.moveaxis(a, time_axis, -1)
        T = a.shape[-1]
        leading = a.shape[:-1]
        C = 1
        for s in leading:
            C *= int(s)
        base = a.reshape(C, T)                 # (C, T)
    if use_conv2d:
        # insert a middle axis H=1 -> (C, 1, T)
        base = base.reshape(base.shape[0], 1, base.shape[1])
    return base


def ensure_list_of_samples(samples, use_conv2d):
    return [normalize_sample_for_model(s, use_conv2d=use_conv2d) for s in samples]


def recursive_collect_arrays(obj):
    '''Collect 1D arrays from nested structure (cams/predicted probs).'''
    found = []
    if obj is None:
        return found
    if isinstance(obj, np.ndarray):
        a = np.squeeze(obj)
        if a.ndim == 1:
            found.append(a)
        else:
            # collapse other axes by mean -> get a 1D time vector
            found.append(np.mean(a, axis=tuple(range(a.ndim - 1))))
        return found
    if isinstance(obj, (list, tuple)):
        for v in obj:
            found.extend(recursive_collect_arrays(v))
        return found
    if isinstance(obj, dict):
        for v in obj.values():
            found.extend(recursive_collect_arrays(v))
        return found
    return found


def align_and_average(cams_list, target_length=None):
    if len(cams_list) == 0:
        return None, None
    lengths = [len(c) for c in cams_list]
    L = target_length or max(lengths)
    if all(l == L for l in lengths):
        stack = np.vstack(cams_list)
    else:
        stack = []
        for c in cams_list:
            old = np.linspace(0, 1, num=len(c))
            new = np.linspace(0, 1, num=L)
            stack.append(np.interp(new, old, c))
        stack = np.vstack(stack)
    return stack.mean(axis=0), stack


def save_array_and_plot(arr, out_path, title=None):
    np.save(out_path + ".npy", arr)
    plt.figure(figsize=(8,3))
    plt.plot(arr)
    plt.title(title or os.path.basename(out_path))
    plt.xlabel("Time (normalized)")
    plt.ylabel("Activation")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path + ".png", bbox_inches='tight', dpi=500)
    plt.close()


# ---------- Main ----------
def process_day(day):
    print(f"\n=== Processing day {day} ===")

    # load test set
    _, _, test_path = conf_train.get_paths(day, small_subsets=USE_SMALL_SUBSETS)
    print("Loading test data from:", test_path)
    df_test = utils.load_data(test_path)
    data_dict = utils.build_data_dict(df_test=df_test)
    _, _, _, _, X_test, y_test = utils._check_data_dict(data=data_dict, require_test=True, only_check_test=True)
    X = utils._sanitize_np(X_test)
    y_true = np.asarray(y_test).astype(int).ravel()
    names = df_test["dish_well"].tolist()
    idxs_to_run = list(range(len(names)))
    
    # prepare output dir for the day
    out_day_dir = os.path.join(OUTPUT_BASE, f"day_{day}")
    os.makedirs(out_day_dir, exist_ok=True)

    # prepare raw samples list
    raw_samples = []
    for i in range(X.shape[0]):
        raw_samples.append(np.asarray(X[i]))
    raw_samples = [raw_samples[i] for i in idxs_to_run]
    data_labels = [int(y_true[i]) for i in idxs_to_run]
    data_names = [names[i] for i in idxs_to_run]
    print(f"Prepared {len(data_names)} samples for day {day}.")

    aggregated_maps = {}

    for model_name in MODELS_TO_RUN:
        print(f"\n--- Model: {model_name} ---")
        model_out_dir = os.path.join(out_day_dir, model_name)
        os.makedirs(model_out_dir, exist_ok=True)

        # load the model
        if model_name == 'LSTMFCN':
            model_path = os.path.join(conf_train.output_model_base_dir, f"best_lstmfcn_model_{day}Days.pth")
            if not os.path.exists(model_path):
                print("LSTMFCN checkpoint not found:", model_path)
                continue
            try:
                model, threshold, saved_params = utils.load_lstmfcn_from_checkpoint(model_path, TimeSeriesClassifier, device=device)
                model = model.to(device); model.eval()
            except Exception as e:
                print("Failed to load LSTMFCN via helper:", e)
                continue

        elif model_name == 'ConvTran':
            model_path = os.path.join(conf_train.output_model_base_dir, f"best_convtran_model_{day}Days.pkl")
            if not os.path.exists(model_path):
                print("ConvTran checkpoint not found:", model_path)
                continue
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            conf_local = copy.deepcopy(conf_train)
            saved_config = checkpoint.get('config', {})
            for key, value in saved_config.items():
                setattr(conf_local, key, value)
            model = model_factory(conf_local).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            threshold = checkpoint.get('best_threshold', 0.5)
            model.eval()
        else:
            print("Unsupported model:", model_name)
            continue
        
        # Create adapter and put it on device
        adapter_model = ModelAdapter(model).to(device)
        adapter_model.eval()

        # save module names for inspection
        named_modules_file = os.path.join(model_out_dir, "named_modules.txt")
        with open(named_modules_file, "w") as fh:
            for n, _ in model.named_modules():
                fh.write(n + "\n")
        print("Wrote named modules to:", named_modules_file)

        # guess whether to present samples as conv1d or conv2d inputs
        override_layer = TARGET_LAYERS_OVERRIDE.get(model_name)
        use_conv2d = False
        if override_layer:
            use_conv2d = model_has_conv2d(model, override_layer)
        else:
            use_conv2d = model_has_conv2d(model)

        if use_conv2d:
            first_conv2d = None
            first_conv_name = None
            for n, m in model.named_modules():
                if isinstance(m, nn.Conv2d):
                    first_conv2d = m
                    first_conv_name = n
                    break
            if first_conv2d is not None and getattr(first_conv2d, "in_channels", None) == 1:
                print(f"Detected Conv2d layer '{first_conv_name}' with in_channels==1; "
                    "treating samples as 2D (C,T) so model's own `unsqueeze` will create channels.")
                use_conv2d = False

        print(f"use_conv2d={use_conv2d} for model {model_name}")

        # normalize samples accordingly
        data_list = ensure_list_of_samples(raw_samples, use_conv2d=use_conv2d)

        # select or auto-select target layer
        target_layer_name = override_layer
        if target_layer_name is None:
            last = None
            for n, m in model.named_modules():
                if use_conv2d and isinstance(m, torch.nn.Conv2d):
                    last = n
                if (not use_conv2d) and isinstance(m, torch.nn.Conv1d):
                    last = n
            if last is None:
                last = None
                for n, m in model.named_modules():
                    if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                        last = n
            if last is None:
                print("Could not find a convolutional layer automatically. Please set TARGET_LAYERS_OVERRIDE.")
                continue
            target_layer_name = last
        print("Using target layer:", target_layer_name)

        cam_builder = TorchCamBuilder(model=adapter_model, class_names=["no_blasto","blasto"], time_axs=-1)

        if SAVE_PER_SAMPLE_VIS:
            results_dir_for_getcam = os.path.join(model_out_dir, "per_sample_outputs")
        else:
            results_dir_for_getcam = None
            print("No saving of per-sample outputs")

        # Run get_cam
        final_layer_name = "base_model." + target_layer_name
        try:
            cams_dict, predicted_probs_dict, bar_ranges = cam_builder.get_cam(
                data_list=data_list,
                data_labels=data_labels,
                target_classes=[0, 1],
                explainer_types=EXPLAINER_TYPES,
                target_layers=[final_layer_name],
                softmax_final=False,
                data_names=data_names,
                display_output=SAVE_PER_SAMPLE_VIS,
                results_dir_path=results_dir_for_getcam,
                aspect_factor=18
            )
        except Exception as e:
            raise RuntimeError(f"cam_builder.get_cam raised an exception: {e}")


        # -------------------- Simplified aggregation & stratified plotting (UPDATED) --------------------
        print("Collecting CAM arrays for aggregation (simplified to two classes, plus stratified views)...")

        # Build per-class, per-sample lists
        N_samples = len(data_list)
        per_class_per_sample = {'0': [ [] for _ in range(N_samples) ], '1': [ [] for _ in range(N_samples) ]}

        if isinstance(cams_dict, dict):
            for full_key, expl_val in cams_dict.items():
                m = re.search(r'_class(\d+)\b', str(full_key))
                if m:
                    class_id = m.group(1)
                else:
                    m2 = re.search(r'class[_\-]?(\d+)', str(full_key))
                    if m2:
                        class_id = m2.group(1)
                    else:
                        s = str(full_key).lower()
                        if 'blasto' in s:
                            class_id = '1' if 'no' not in s else '0'
                        else:
                            continue
                if class_id not in ('0', '1'):
                    continue

                if isinstance(expl_val, (list, tuple, np.ndarray)):
                    for idx in range(min(N_samples, len(expl_val))):
                        element = expl_val[idx]
                        found = recursive_collect_arrays(element)
                        if len(found) >= 1:
                            per_class_per_sample[class_id][idx].append(found[0])
        else:
            arrays_all = recursive_collect_arrays(cams_dict)
            if arrays_all:
                per_class_per_sample['0'][0].extend(arrays_all)

        def extract_class1_probs(predicted_probs_dict, N_samples):
            if isinstance(predicted_probs_dict, dict):
                for k, v in predicted_probs_dict.items():
                    try:
                        if re.search(r'class1\b', k) or k.lower().endswith('class1'):
                            arrs = recursive_collect_arrays(v)
                            for a in arrs:
                                if isinstance(a, np.ndarray) and a.ndim == 1 and len(a) == N_samples:
                                    return a
                    except Exception:
                        continue
                for k, v in predicted_probs_dict.items():
                    try:
                        if re.search(r'class0\b', k) or k.lower().endswith('class0'):
                            arrs = recursive_collect_arrays(v)
                            for a in arrs:
                                if isinstance(a, np.ndarray) and a.ndim == 1 and len(a) == N_samples:
                                    return 1.0 - a
                    except Exception:
                        continue
            found = recursive_collect_arrays(predicted_probs_dict)
            for a in found:
                if isinstance(a, np.ndarray) and a.ndim == 1 and len(a) == N_samples:
                    return a
            return None

        pick = extract_class1_probs(predicted_probs_dict, N_samples)

        if pick is None or len(pick) != N_samples:
            print("Computing predicted probabilities via model fallback (stratified)...")
            pick_list = []
            for sample in data_list:
                arr = np.asarray(sample)
                if arr.ndim == 3 and use_conv2d:
                    t = torch.tensor(arr[np.newaxis, :, :, :], dtype=torch.float32).to(device)
                elif arr.ndim == 2 and (not use_conv2d):
                    t = torch.tensor(arr[np.newaxis, :, :], dtype=torch.float32).to(device)
                elif arr.ndim == 1:
                    t = torch.tensor(arr[np.newaxis, :, None], dtype=torch.float32).to(device)
                else:
                    t = torch.tensor(arr[np.newaxis, ...], dtype=torch.float32).to(device)
                try:
                    out = model(t)
                except Exception:
                    try:
                        out = adapter_model(t.permute(0, 2, 1))
                    except Exception:
                        out = None
                prob = 0.0
                if out is not None:
                    try:
                        prob = float(torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy().ravel()[0])
                    except Exception:
                        try:
                            prob = float(torch.sigmoid(out).detach().cpu().numpy().ravel()[0])
                        except Exception:
                            prob = 0.0
                pick_list.append(prob)
            pick = np.array(pick_list)

        try:
            thresh = float(threshold)
        except Exception:
            thresh = 0.5
        pred_labels = (pick >= thresh).astype(int)

        # -------------------- Prediction Summary CSV --------------------
        def compute_topk_peaks_for_sample(idx, per_class_per_sample, top_k=5, window_frac=0.02):
            arrs = []
            arrs.extend(per_class_per_sample.get('0', [])[idx] if idx < len(per_class_per_sample.get('0', [])) else [])
            arrs.extend(per_class_per_sample.get('1', [])[idx] if idx < len(per_class_per_sample.get('1', [])) else [])
            if not arrs:
                return [''] * top_k

            avg, stack = align_and_average(arrs)
            if avg is None:
                return [''] * top_k

            importance = np.asarray(avg).copy()
            if np.ptp(importance) > 1e-9:
                importance = (importance - importance.min()) / (importance.max() - importance.min())

            T = len(importance)
            if T == 0:
                return [''] * top_k

            window_frames = max(1, int(round(window_frac * T)))
            inds_sorted = np.argsort(importance)[::-1]

            selected = []
            for ind in inds_sorted:
                if len(selected) >= top_k:
                    break
                ind = int(ind)
                if any(abs(ind - s) <= window_frames for s in selected):
                    continue
                selected.append(ind)

            peak_hours = [f"{(idx_peak * ACQUISITION_STEP_HOURS):.2f}" for idx_peak in selected]
            while len(peak_hours) < top_k:
                peak_hours.append('')
            return peak_hours

        summary_csv_path = os.path.join(model_out_dir, f"summary_prediction_day{day}_{model_name}.csv")
        peak_cols = [f"peak{i+1}_h" for i in range(5)]
        with open(summary_csv_path, "w", newline="") as csvf:
            writer = csv.writer(csvf)
            writer.writerow(["dish_well", "true_label", "pred_label", "prob_class1"] + peak_cols)

            for i in range(N_samples):
                dish = data_names[i] if i < len(data_names) else f"idx_{i}"
                true_lbl = int(y_true[i])
                pred_lbl = int(pred_labels[i])
                prob = float(pick[i]) if (pick is not None and i < len(pick)) else float('nan')
                peaks = compute_topk_peaks_for_sample(i, per_class_per_sample, top_k=5, window_frac=0.02)
                writer.writerow([dish, true_lbl, pred_lbl, prob] + peaks)

        print("Wrote prediction summary CSV with top-5 peaks:", summary_csv_path)

        # -------------------- Prepare Stratified Aggregation --------------------
        def flatten_for_indices(per_sample_lists, indices):
            out = []
            for i in indices:
                items = per_sample_lists[i]
                if items:
                    out.extend(items)
            return out

        model_aggregated = {}
        global_for_norm = []

        for clsid in ['0', '1']:
            indices_with_cam = [i for i in range(N_samples) if len(per_class_per_sample[clsid][i]) > 0]
            all_arrays = flatten_for_indices(per_class_per_sample[clsid], indices_with_cam)

            idx_pred_cls = [i for i in range(N_samples) if pred_labels[i] == int(clsid)]
            idx_pred_cls_correct = [i for i in idx_pred_cls if int(y_true[i]) == int(clsid)]
            idx_pred_cls_wrong = [i for i in idx_pred_cls if int(y_true[i]) != int(clsid)]

            right_arrays = flatten_for_indices(per_class_per_sample[clsid], idx_pred_cls_correct)
            wrong_arrays = flatten_for_indices(per_class_per_sample[clsid], idx_pred_cls_wrong)

            for arr_list in (all_arrays, right_arrays, wrong_arrays):
                if arr_list:
                    _, st = align_and_average(arr_list)
                    global_for_norm.append(st)
            
            # Map simplified keys to descriptive titles for display
            cls_label_text = 'Blasto' if clsid == '1' else 'No-Blasto'
            model_aggregated[clsid] = {
                'all': {
                    'arrays': all_arrays, 
                    'indices': indices_with_cam,
                    'title': f"All Predictions (Predicted as {cls_label_text})"
                },
                'right': {
                    'arrays': right_arrays, 
                    'indices': idx_pred_cls_correct,
                    'title': f"Correctly Predicted as {cls_label_text} (True {'Positives' if clsid=='1' else 'Negatives'})"
                },
                'wrong': {
                    'arrays': wrong_arrays, 
                    'indices': idx_pred_cls_wrong,
                    'title': f"Incorrectly Predicted as {cls_label_text} (False {'Positives' if clsid=='1' else 'Negatives'})"
                }
            }

        if global_for_norm:
            combined_all = np.vstack(global_for_norm)
            global_min = float(np.min(combined_all))
            global_max = float(np.max(combined_all))
            if np.isclose(global_max, global_min):
                global_min, global_max = 0.0, 1.0
        else:
            global_min, global_max = 0.0, 1.0

        def compute_avg_stack(arrs):
            if not arrs:
                return None, None
            return align_and_average(arrs)

        # Plot vertically (3 rows x 1 col)
        for clsid, info in model_aggregated.items():
            # Define parts order
            parts = [('all', info['all']), ('right', info['right']), ('wrong', info['wrong'])]
            
            fig, axes = plt.subplots(3, 1, figsize=(10, 15), squeeze=False)
            axes = axes[:, 0]
            T_used = None
            
            cls_name_text = 'blasto' if clsid == '1' else 'no_blasto'

            for ax_idx, (label_part, info_part) in enumerate(parts):
                ax = axes[ax_idx]
                arrs = info_part['arrays']
                indices = info_part.get('indices', [])
                plot_title = info_part.get('title', label_part)
                
                # --- FIXED Y-LIMITS ---
                ax.set_ylim(-0.05, 1.05)

                if not arrs:
                    ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=21)
                    # Title
                    ax.set_title(f"{plot_title} (n_maps=0)", fontsize=21, pad=8)
                    ax.set_axis_off()
                    continue

                avg, stack = compute_avg_stack(arrs)
                if avg is None:
                    ax.text(0.5, 0.5, 'No valid maps', ha='center', va='center', fontsize=21)
                    ax.set_axis_off()
                    continue

                if T_used is None:
                    T_used = avg.shape[0]

                denom = (global_max - global_min) if (global_max - global_min) != 0 else 1.0
                avg_norm = (avg - global_min) / denom
                stack_norm = (stack - global_min) / denom
                std_norm = stack_norm.std(axis=0)

                hours = np.arange(len(avg_norm)) * ACQUISITION_STEP_HOURS
                ax.plot(hours, avg_norm, linewidth=2.5, color='#1f77b4') # thicker line
                ax.fill_between(hours, np.clip(avg_norm - std_norm, 0, 1), np.clip(avg_norm + std_norm, 0, 1), 
                                alpha=0.3, color='#1f77b4')
                ax.grid(alpha=0.3)
                
                # Axis labels
                ax.set_ylabel('Normalized Activation (0-1)', fontsize=16)
                
                # --- X-LABEL ONLY ON LAST PLOT ---
                if ax_idx == len(parts) - 1:
                    ax.set_xlabel('Time (hours)', fontsize=18, labelpad=15)
                else:
                    ax.set_xlabel('')

                # --- 25 TICKS (FIRST AND LAST INCLUDED) ---
                T = len(avg_norm)
                n_ticks_desired = 25
                if T <= 1:
                    tick_indices = np.array([0])
                else:
                    # linspace ensures we get start, end, and even spacing
                    tick_indices = np.linspace(0, T - 1, n_ticks_desired, dtype=int)
                    # unique checks just in case T < 25
                    tick_indices = np.unique(tick_indices)
                    
                xtick_positions = tick_indices * ACQUISITION_STEP_HOURS
                xtick_labels = [f"{pos:.1f}h" for pos in xtick_positions]
                
                ax.set_xticks(xtick_positions)
                ax.set_xticklabels(xtick_labels, rotation=45, fontsize=14)
                ax.tick_params(axis='y', labelsize=14)

                # Peaks
                top_k = 5
                importance = avg_norm.copy()
                window_frac = 0.02
                window_frames = max(1, int(round(window_frac * T)))
                inds_sorted = np.argsort(importance)[::-1]
                selected = []
                for ind in inds_sorted:
                    if len(selected) >= top_k:
                        break
                    if any(abs(int(ind) - s) <= window_frames for s in selected):
                        continue
                    selected.append(int(ind))

                ymin, ymax = -0.05, 1.05  # use our fixed limits
                half_win = max(1, window_frames) / 2.0
                for rank, idx_peak in enumerate(selected, start=1):
                    start_idx = max(0, int(idx_peak - np.floor(half_win)))
                    end_idx = min(T - 1, int(idx_peak + np.floor(half_win)))
                    start_h = start_idx * ACQUISITION_STEP_HOURS
                    end_h = (end_idx + 1) * ACQUISITION_STEP_HOURS
                    ax.axvspan(start_h, end_h, color='gold', alpha=0.3, linewidth=0)
                    ax.axvline(x=idx_peak * ACQUISITION_STEP_HOURS, color='darkorange', linestyle='--', linewidth=1.5)
                    
                    # Position peak text relative to fixed ylim
                    y_offset = (rank - 1) * 0.1 * (ymax - ymin)
                    # Peak text
                    ax.text(idx_peak * ACQUISITION_STEP_HOURS, ymax*0.95 - y_offset,
                            f"{idx_peak * ACQUISITION_STEP_HOURS:.2f}h", ha='center', va='top', fontsize=14,
                            color='black', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

                # Descriptive Subplot Title
                n_maps = int(stack.shape[0]) if stack is not None else 0
                n_unique_samples = len(indices)
                # no new line in title but tab for readability
                ax.set_title(f"{plot_title} (n_maps={n_maps})", fontsize=21, pad=8)

            # Figure Suptitle
            total_embryo_class = len(model_aggregated[clsid]['right']['indices']) + len(model_aggregated[clsid]['wrong']['indices'])
            fig.suptitle(f"{model_name} - Grad-CAM - class {cls_name_text} — class samples: {total_embryo_class}", fontsize=21, y=0.98)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.96], h_pad=3.0)
            outpng = os.path.join(model_out_dir, f"global_class_{clsid}_stratified.png")
            plt.savefig(outpng, bbox_inches='tight', dpi=500)
            plt.close()
            print(f"Saved stratified (vertical) plot for class {clsid}:", outpng)

            # Save avg/stack .npy files
            for label_part, _info in parts:
                arrs = model_aggregated[clsid].get(label_part, {}).get('arrays', [])
                avg, stack = compute_avg_stack(arrs)
                if avg is not None:
                    model_aggregated[clsid].setdefault(label_part, {})
                    model_aggregated[clsid][label_part]['avg'] = avg
                    model_aggregated[clsid][label_part]['stack'] = stack
                    model_aggregated[clsid][label_part]['n_samples'] = int(stack.shape[0]) if stack is not None else 0
                    np.save(os.path.join(model_out_dir, f"class_{clsid}_{label_part}_avg.npy"), avg)
                    if stack is not None:
                        np.save(os.path.join(model_out_dir, f"class_{clsid}_{label_part}_stack.npy"), stack)

        aggregated_maps[model_name] = model_aggregated

        # -------------------- Prob-weighted maps (Legacy) --------------------
        found_probs = recursive_collect_arrays(predicted_probs_dict)
        pick = None
        if found_probs:
            for a in found_probs:
                if a.ndim == 1 and len(a) == len(data_list):
                    pick = a
                    break
        if pick is None or len(pick) != len(data_list):
            print("Computing predicted probabilities via model fallback...")
            pick_list = []
            for sample in data_list:
                arr = np.asarray(sample)
                if arr.ndim == 2 and use_conv2d:
                    t = torch.tensor(arr[np.newaxis, :, :, :], dtype=torch.float32).to(device)
                elif arr.ndim == 2 and (not use_conv2d):
                    t = torch.tensor(arr[np.newaxis, :, :], dtype=torch.float32).to(device)
                elif arr.ndim == 1:
                    t = torch.tensor(arr[np.newaxis, :, None], dtype=torch.float32).to(device)
                else:
                    t = torch.tensor(arr[np.newaxis, ...], dtype=torch.float32).to(device)
                try:
                    out = model(t)
                except Exception:
                    try:
                        out = adapter_model(t.permute(0, 2, 1))
                    except Exception:
                        out = None
                prob = 0.0
                if out is not None:
                    try:
                        prob = float(torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy().ravel()[0])
                    except Exception:
                        try:
                            prob = float(torch.sigmoid(out).detach().cpu().numpy().ravel()[0])
                        except Exception:
                            prob = 0.0
                pick_list.append(prob)
            pick = np.array(pick_list)

        for expl_key, expl_res in model_aggregated.items():
            if 'all' in expl_res and expl_res['all']['stack'] is not None:
                stack = expl_res['all']['stack']
                if stack.shape[0] == len(pick):
                    weights = pick[:, None]
                    weighted = (stack * weights).sum(axis=0) / (weights.sum() + 1e-9)
                    outprefix = os.path.join(model_out_dir, f"global_{expl_key}_prob_weighted")
                    save_array_and_plot(weighted, outprefix, title=f"{model_name} {expl_key} prob-weighted (day {day})")
                    print("Saved prob-weighted map:", outprefix + ".npy/.png")

    # --- robust summary writer ---
    summary = {}
    for mname, exps in aggregated_maps.items():
        summary[mname] = {}
        if not isinstance(exps, dict) or len(exps) == 0:
            continue

        sample_val = next(iter(exps.values()))
        if isinstance(sample_val, dict) and any(k in sample_val for k in ('avg', 'stack', 'n_samples')):
            for clsid, info in exps.items():
                n = None
                if isinstance(info, dict):
                    n = info.get('n_samples')
                    if n is None and isinstance(info.get('stack'), np.ndarray):
                        try:
                            n = int(info['stack'].shape[0])
                        except Exception:
                            n = None
                summary[mname][clsid] = {'n_samples': n}
        else:
            for expl_name, classes in exps.items():
                summary[mname][expl_name] = {}
                if not isinstance(classes, dict):
                    continue
                for clsid, info in classes.items():
                    n = None
                    if isinstance(info, dict):
                        n = info.get('n_samples')
                        if n is None and isinstance(info.get('stack'), np.ndarray):
                            try:
                                n = int(info['stack'].shape[0])
                            except Exception:
                                n = None
                    summary[mname][expl_name][clsid] = {'n_samples': n}

    with open(os.path.join(out_day_dir, "aggregated_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Finished day {day}. Outputs in {out_day_dir}")
    return aggregated_maps


def main():
    all_maps = {}
    for d in DAYS:
        try:
            all_maps[d] = process_day(d)
        except Exception as e:
            print("Error processing day", d, ":", e)
    print("All done. Outputs under:", OUTPUT_BASE)

if __name__ == '__main__':
    main()