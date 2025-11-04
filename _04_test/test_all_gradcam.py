#!/usr/bin/env python3
"""
test_with_GradCAM_fixed2.py

Fixes:
 - Avoid 5D Conv2d input errors by canonicalizing each sample depending on model conv type
 - Avoid passing results_dir_path=None to `cam_builder.get_cam()`. If SAVE_PER_SAMPLE_VIS=False,
   use a temporary directory and delete it after `get_cam` completes, preventing persistent files.

Improvements in this version:
 - Simplified aggregation: produce only TWO averaged activation maps (one per class: 0 and 1)
   and their stacks (.npy saved). This removes the many redundant outputs.
 - Improved plotting: mean + shaded std area, normalization to [0,1] (global across classes for
   comparability), descriptive titles, and x-axis in hours (acquisition step = 15 minutes = 0.25 h).

Usage: drop in your cellPIV repo and run:
    python _04_test/test_with_GradCAM_fixed2.py
"""

import os
import sys
import copy
import json
import shutil
import tempfile
from collections import defaultdict

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
DAYS = [1, 3, 5]                         # days to process
MODELS_TO_RUN = ['LSTMFCN', 'ConvTran']  # choose subset
OUTPUT_BASE = os.path.join(current_dir, "GRADCAM_batch_outputs")
os.makedirs(OUTPUT_BASE, exist_ok=True)

EXPLAINER_TYPES = ["Grad-CAM"]     # or ["Grad-CAM","HiResCAM"]
TARGET_LAYERS_OVERRIDE = {'LSTMFCN': None, 'ConvTran': None}  # set to layer name if needed
SAVE_PER_SAMPLE_VIS = False        # If True -> files are kept; if False -> use tempdir and delete
ACQUISITION_STEP_HOURS = 0.25     # 15 minutes = 0.25 hours
# ------------------------------------------------

# ---------- Helpers ----------
def model_has_conv2d(model, layer_name=None):
    """Return True if model (or the specified layer module) contains Conv2d."""
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
    """
    Convert arbitrary-sample numpy array -> appropriate shape depending on model:
      - if use_conv2d True: return shape (C, H=1, W=T)  (so batch -> (N,C,1,T) valid 4D for Conv2d)
      - otherwise return shape (C, T)
    Strategy:
      - Move the largest axis to the last position (assume time)
      - Collapse all leading dims into channels
      - If conv2d requested, insert a middle spatial axis H=1
    """
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
    """Collect 1D arrays from nested structure (cams/predicted probs)."""
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
    plt.savefig(out_path + ".png", bbox_inches='tight', dpi=200)
    plt.close()


# ---------- Main ----------
def process_day(day):
    print(f"\n=== Processing day {day} ===")

    # load test set
    _, _, test_path = conf_train.get_paths(day)
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
        # if a target layer override is provided, check that layer's type; else check whole model for Conv2d
        override_layer = TARGET_LAYERS_OVERRIDE.get(model_name)
        use_conv2d = False
        if override_layer:
            use_conv2d = model_has_conv2d(model, override_layer)
        else:
            # if the model contains Conv2d anywhere, prefer conv2d representation
            use_conv2d = model_has_conv2d(model)

        # Special-case detection: some Conv2d-based architectures (ConvTran) expect
        # inputs shaped (C, T) and call `x = x.unsqueeze(1)` inside forward.
        # If the first Conv2d has in_channels == 1, treat the data as 2D (C, T).
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
            # pick last conv layer
            # prefer conv2d layer name if using conv2d, else conv1d
            last = None
            for n, m in model.named_modules():
                if use_conv2d and isinstance(m, torch.nn.Conv2d):
                    last = n
                if (not use_conv2d) and isinstance(m, torch.nn.Conv1d):
                    last = n
            if last is None:
                # as fallback choose any conv module present
                last = None
                for n, m in model.named_modules():
                    if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                        last = n
            if last is None:
                print("Could not find a convolutional layer automatically. Please set TARGET_LAYERS_OVERRIDE.")
                continue
            target_layer_name = last
        print("Using target layer:", target_layer_name)

        # Prepare TorchCamBuilder. Important: set time_axs = -1 (time last axis)
        # If we used conv2d representation, our sample shape is (C,1,T) -> time is last axis -> time_axs=-1
        cam_builder = TorchCamBuilder(model=adapter_model, class_names=["no_blasto","blasto"], time_axs=-1)

        # Prepare results dir for get_cam:
        if SAVE_PER_SAMPLE_VIS:
            results_dir_for_getcam = os.path.join(model_out_dir, "per_sample_outputs")
        else:
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

        # -------------------- Simplified aggregation & improved plotting --------------------
        import re

        print("Collecting CAM arrays for aggregation (simplified to two classes)...")
        cams_by_explainer_class = defaultdict(lambda: defaultdict(list))

        if isinstance(cams_dict, dict):
            for full_key, expl_val in cams_dict.items():
                # full_key examples:
                #  'Grad-CAM_base_model.embed_layer.3_class0'
                #  'Grad-CAM_base_model.embed_layer.3_class1'
                # estrai nome explainer (es. 'Grad-CAM') e classe ('0' o '1')
                explainer_name = None
                class_id = None

                # explainer name = part before the first underscore (common in your examples)
                if isinstance(full_key, str) and "_" in full_key:
                    explainer_name = full_key.split("_", 1)[0]
                else:
                    explainer_name = str(full_key)

                # cerca pattern _classX (preferito)
                m = re.search(r'_class(\d+)\b', str(full_key))
                if m:
                    class_id = m.group(1)
                else:
                    # fallback: cerca 'classX' altrove
                    m2 = re.search(r'class[_\-]?(\d+)', str(full_key))
                    if m2:
                        class_id = m2.group(1)
                    else:
                        # ultima possibilità: usa parole chiave 'blasto' / 'no'
                        s = str(full_key).lower()
                        if 'blasto' in s:
                            class_id = '1' if 'no' not in s else '0'

                # estrai tutte le array 1D presenti in expl_val (può essere list/tuple/dict/ndarray)
                arrays = []
                if isinstance(expl_val, (list, tuple)):
                    for item in expl_val:
                        arrays.extend(recursive_collect_arrays(item))
                elif isinstance(expl_val, dict):
                    # vecchio formato annidato: scorre i valori interni
                    for v in expl_val.values():
                        arrays.extend(recursive_collect_arrays(v))
                else:
                    arrays.extend(recursive_collect_arrays(expl_val))

                # Se non siamo riusciti a determinare la classe, ignora (semplificazione voluta)
                if class_id is None:
                    continue

                # aggiungi alle liste per explainer/class
                cams_by_explainer_class[explainer_name][class_id].extend(arrays)

        else:
            # se non è dict, prova a estrarre tutto ma senza assegnare classi
            arrays_all = recursive_collect_arrays(cams_dict)
            if arrays_all:
                cams_by_explainer_class['generic']['unknown'] = arrays_all

        # Combine across explainers into per-class lists (only 0 and 1)
        class_arrays = {'0': [], '1': []}
        for expl_vals in cams_by_explainer_class.values():
            for clsid in ['0', '1']:
                class_arrays[clsid].extend(expl_vals.get(clsid, []))

        # Compute averages, stacks and save only two .npy (avg + stack) and improved plots
        model_aggregated = {}
        # gather combined values to compute global normalization
        combined_for_norm = []
        for clsid in ['0', '1']:
            if class_arrays[clsid]:
                _, st = align_and_average(class_arrays[clsid])
                combined_for_norm.append(st)
        if combined_for_norm:
            combined_all = np.vstack(combined_for_norm)
            global_min = float(np.min(combined_all))
            global_max = float(np.max(combined_all))
            if np.isclose(global_max, global_min):
                global_min, global_max = 0.0, 1.0
        else:
            global_min, global_max = 0.0, 1.0

        for clsid in ['0', '1']:
            arrs = class_arrays[clsid]
            if not arrs:
                print(f"No activation maps found for class {clsid}; skipping.")
                continue
            avg, stack = align_and_average(arrs)
            n_maps = stack.shape[0]
            model_aggregated[clsid] = {'avg': avg, 'stack': stack, 'n_samples': n_maps}

            # Save stacks and avg as .npy (kept minimal: only per-class files)
            np.save(os.path.join(model_out_dir, f"class_{clsid}_avg.npy"), avg)
            np.save(os.path.join(model_out_dir, f"class_{clsid}_stack.npy"), stack)

        # Produce improved plots (normalized globally across classes to keep comparability)
        for clsid, info in model_aggregated.items():
            avg = info['avg']
            stack = info['stack']
            n_maps = info['n_samples']
            L = avg.shape[0]
            # normalize to [0,1] using global min/max for comparability across classes
            denom = (global_max - global_min) if (global_max - global_min) != 0 else 1.0
            avg_norm = (avg - global_min) / denom
            stack_norm = (stack - global_min) / denom
            std_norm = stack_norm.std(axis=0)

            hours = np.arange(L) * ACQUISITION_STEP_HOURS

            plt.figure(figsize=(12, 4))
            plt.plot(hours, avg_norm, linewidth=1.5, label='Mean activation')
            # shaded std area
            plt.fill_between(hours, np.clip(avg_norm - std_norm, 0, 1), np.clip(avg_norm + std_norm, 0, 1),
                            alpha=0.25, label='±1 std')
            plt.grid(alpha=0.25)
            plt.xlabel('Time (hours)')
            plt.ylabel('Normalized activation (0-1)')

            # xticks: choose step so we have at most 30 ticks
            max_ticks = 30
            if L <= 1:
                tick_indices = np.array([0])
            else:
                step_frames = max(1, int(np.ceil(L / max_ticks)))
                tick_indices = np.arange(0, L, step_frames)
                if tick_indices[-1] != (L - 1):
                    # ensure last tick is the final point
                    tick_indices = np.concatenate([tick_indices, np.array([L - 1])])
            xtick_positions = tick_indices * ACQUISITION_STEP_HOURS
            xtick_labels = [f"{pos:.2f}h" for pos in xtick_positions]
            plt.xticks(xtick_positions, xtick_labels, rotation=45)

            # Identify top-k peaks on the averaged (unnormalized) or normalized curve
            top_k = 5
            # use the normalized average as importance score
            importance = avg_norm.copy()
            # select peaks by descending order while avoiding near-duplicates: require min distance = window_frames
            window_frac = 0.02  # highlight window as fraction of series length (2% of length)
            window_frames = max(1, int(round(window_frac * L)))
            inds_sorted = np.argsort(importance)[::-1]  # descending
            selected = []
            for ind in inds_sorted:
                if len(selected) >= top_k:
                    break
                # skip if too close to any already selected
                if any(abs(int(ind) - s) <= window_frames for s in selected):
                    continue
                selected.append(int(ind))

            # draw highlights in HOURS coordinates (x axis is in hours)
            ymin, ymax = plt.ylim()
            half_win = max(1, window_frames) / 2.0
            for rank, idx_peak in enumerate(selected, start=1):
                start_idx = max(0, int(idx_peak - np.floor(half_win)))
                end_idx = min(L - 1, int(idx_peak + np.floor(half_win)))
                start_h = start_idx * ACQUISITION_STEP_HOURS
                end_h = (end_idx + 1) * ACQUISITION_STEP_HOURS  # inclusive
                plt.axvspan(start_h, end_h, color='yellow', alpha=0.25, linewidth=0)
                plt.axvline(x=idx_peak * ACQUISITION_STEP_HOURS, color='orange', linestyle='--', linewidth=1.0)
                # annotate normalized score slightly below top of plot, stagger by rank to reduce overlap
                y_offset = (rank - 1) * 0.08 * (ymax - ymin)
                # annotate the started time in hours
                plt.text(idx_peak * ACQUISITION_STEP_HOURS, ymax*0.9 - y_offset,
                        f"{idx_peak * ACQUISITION_STEP_HOURS:.2f}h", ha='center', va='top', fontsize=8,
                        color='black', backgroundcolor='white')

            title = (f"{model_name} - Grad-CAM average for class {clsid} "
                    f"({'blasto' if clsid=='1' else 'no_blasto'}) — {n_maps} maps — "
                    f"Top {len(selected)} peaks highlighted")
            plt.title(title)
            plt.tight_layout()
            outpng = os.path.join(model_out_dir, f"global_class_{clsid}_avg_normalized.png")
            plt.savefig(outpng, bbox_inches='tight', dpi=200)
            plt.close()
            print(f"Saved simplified normalized plot for class {clsid}:", outpng)

        aggregated_maps[model_name] = model_aggregated

        # -------------------- End simplified aggregation --------------------

        # probability-weighted average (kept original behavior)
        found_probs = recursive_collect_arrays(predicted_probs_dict)
        pick = None
        if found_probs:
            for a in found_probs:
                if a.ndim == 1 and len(a) == len(data_list):
                    pick = a
                    break
        if pick is None or len(pick) != len(data_list):
            # fallback: compute using the model
            print("Computing predicted probabilities via model fallback...")
            pick_list = []
            for sample in data_list:
                arr = np.asarray(sample)
                # convert to tensor; keep shape logic coherent:
                if arr.ndim == 2 and use_conv2d:
                    t = torch.tensor(arr[np.newaxis, :, :, :], dtype=torch.float32).to(device)  # (1,C,H,W)
                elif arr.ndim == 2 and (not use_conv2d):
                    t = torch.tensor(arr[np.newaxis, :, :], dtype=torch.float32).to(device)  # (1,C,T)
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

        # generate probability-weighted maps if stack sizes match (this part is optional and will be
        # skipped if our simplified stacks are not compatible in shape)
        for expl_key, expl_res in model_aggregated.items():
            if 'all' in expl_res and expl_res['all']['stack'] is not None:
                stack = expl_res['all']['stack']
                if stack.shape[0] == len(pick):
                    weights = pick[:, None]
                    weighted = (stack * weights).sum(axis=0) / (weights.sum() + 1e-9)
                    outprefix = os.path.join(model_out_dir, f"global_{expl_key}_prob_weighted")
                    save_array_and_plot(weighted, outprefix, title=f"{model_name} {expl_key} prob-weighted (day {day})")
                    print("Saved prob-weighted map:", outprefix + ".npy/.png")

    # --- robust summary writer (sostituire il blocco precedente) ---
    summary = {}
    for mname, exps in aggregated_maps.items():
        summary[mname] = {}
        if not isinstance(exps, dict) or len(exps) == 0:
            continue

        # inspect a sample value to detect format
        sample_val = next(iter(exps.values()))
        # caso 1: exps è { classid: { 'avg':..., 'stack':..., 'n_samples':... }, ... }
        if isinstance(sample_val, dict) and any(k in sample_val for k in ('avg', 'stack', 'n_samples')):
            for clsid, info in exps.items():
                n = None
                if isinstance(info, dict):
                    # prefer 'n_samples' se presente, altrimenti inferisci da 'stack'
                    n = info.get('n_samples')
                    if n is None and isinstance(info.get('stack'), np.ndarray):
                        try:
                            n = int(info['stack'].shape[0])
                        except Exception:
                            n = None
                summary[mname][clsid] = {'n_samples': n}
        else:
            # caso 2: exps è { explainer_name: { classid: { ... }, ... }, ... }
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
    # --- end summary writer ---

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
