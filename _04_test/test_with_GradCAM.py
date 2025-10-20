#!/usr/bin/env python3
"""
test_with_GradCAM_fixed2.py

Fixes:
 - Avoid 5D Conv2d input errors by canonicalizing each sample depending on model conv type
 - Avoid passing results_dir_path=None to `cam_builder.get_cam()`. If SAVE_PER_SAMPLE_VIS=False,
   use a temporary directory and delete it after `get_cam` completes, preventing persistent files.

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
DAYS = [1]                         # days to process
MODELS_TO_RUN = ['ConvTran']  # choose subset
OUTPUT_BASE = os.path.join(current_dir, "GRADCAM_batch_outputs")
os.makedirs(OUTPUT_BASE, exist_ok=True)

SPECIFIC_VIDEO = None              # e.g. "D2013.03.09_S0695_I141_1" or None
EXPLAINER_TYPES = ["Grad-CAM"]     # or ["Grad-CAM","HiResCAM"]
TARGET_LAYERS_OVERRIDE = {'LSTMFCN': None, 'ConvTran': None}  # set to layer name if needed
SAVE_PER_SAMPLE_VIS = False        # If True -> files are kept; if False -> use tempdir and delete
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
    out_day_dir = os.path.join(OUTPUT_BASE, f"day_{day}")
    os.makedirs(out_day_dir, exist_ok=True)

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
    if SPECIFIC_VIDEO:
        if SPECIFIC_VIDEO in names:
            idxs_to_run = [names.index(SPECIFIC_VIDEO)]
        else:
            print("WARNING: specific video not found; running all.")

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
            results_dir_for_getcam = model_out_dir
        else:
            # create a temporary dir and pass its path (signal_grad_cam wants a path-type)
            temp_dir = tempfile.mkdtemp(prefix="gradcam_tmp_")
            results_dir_for_getcam = temp_dir
            print("Using temporary results dir (will be deleted):", temp_dir)

        # Run get_cam (it will write to results_dir_for_getcam)
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
                results_dir_path=results_dir_for_getcam,
                aspect_factor=18
            )
        except Exception as e:
            # cleanup temp dir if created
            if not SAVE_PER_SAMPLE_VIS and os.path.exists(results_dir_for_getcam):
                try:
                    shutil.rmtree(results_dir_for_getcam)
                except Exception:
                    pass
            raise RuntimeError(f"cam_builder.get_cam raised an exception: {e}")

        # If we used temporary folder and user doesn't want per-sample outputs, delete it now
        if (not SAVE_PER_SAMPLE_VIS) and os.path.exists(results_dir_for_getcam):
            try:
                shutil.rmtree(results_dir_for_getcam)
                print("Deleted temporary results dir:", results_dir_for_getcam)
            except Exception as e:
                print("Warning: could not delete temp results dir:", e)

        # Optionally create an aggregated per-sample summary image using builder helper (only if desired)
        if SAVE_PER_SAMPLE_VIS:
            try:
                cam_builder.single_channel_output_display(
                    data_list=data_list,
                    data_labels=data_labels,
                    predicted_probs_dict=predicted_probs_dict,
                    cams_dict=cams_dict,
                    explainer_types=EXPLAINER_TYPES[0],
                    target_classes=[0,1],
                    target_layers=[final_layer_name],
                    data_names=data_names,
                    fig_size=(8,4),
                    grid_instructions=(1,1),
                    bar_ranges_dict=bar_ranges,
                    results_dir_path=model_out_dir,
                    dt=10,
                    line_width=0.5,
                    marker_width=30,
                    axes_names=("Time Steps","Signal")
                )
            except Exception as e:
                print("single_channel_output_display failed:", e)

        # -- aggregation: collect CAM arrays from cams_dict and produce global maps --
        print("Collecting CAM arrays for aggregation...")
        cams_by_explainer_class = defaultdict(lambda: defaultdict(list))
        if isinstance(cams_dict, dict):
            for explainer_key, expl_val in cams_dict.items():
                if isinstance(expl_val, dict):
                    for k1, v1 in expl_val.items():
                        if isinstance(v1, dict):
                            for k2, v2 in v1.items():
                                arrays = recursive_collect_arrays(v2)
                                assigned = None
                                try:
                                    assigned = int(k2)
                                except Exception:
                                    if isinstance(k2, str) and 'blast' in k2.lower():
                                        assigned = 1 if 'no' not in k2.lower() else 0
                                if assigned is None:
                                    cams_by_explainer_class[explainer_key]['unknown'].extend(arrays)
                                else:
                                    cams_by_explainer_class[explainer_key][str(assigned)].extend(arrays)
                        else:
                            arrays = recursive_collect_arrays(v1)
                            cams_by_explainer_class[explainer_key]['unknown'].extend(arrays)
                else:
                    arrays = recursive_collect_arrays(expl_val)
                    cams_by_explainer_class[explainer_key]['unknown'].extend(arrays)
        else:
            arrays_all = recursive_collect_arrays(cams_dict)
            if arrays_all:
                cams_by_explainer_class['unknown']['unknown'] = arrays_all

        # compute averages and save
        model_aggregated = {}
        for expl_key, classes in cams_by_explainer_class.items():
            model_aggregated[expl_key] = {}
            for cls_key, arrs in classes.items():
                if not arrs:
                    continue
                avg, stack = align_and_average(arrs)
                model_aggregated[expl_key][cls_key] = {'avg': avg, 'stack': stack, 'n_samples': stack.shape[0]}
                outprefix = os.path.join(model_out_dir, f"global_{expl_key}_class{cls_key}")
                save_array_and_plot(avg, outprefix, title=f"{model_name} {expl_key} class {cls_key} (day {day})")
                print("Saved global map:", outprefix + ".npy/.png (n_samples={})".format(stack.shape[0]))

            # combined across classes
            combined = []
            for arrs in classes.values():
                combined.extend(arrs)
            if combined:
                avg_all, stack_all = align_and_average(combined)
                outprefix = os.path.join(model_out_dir, f"global_{expl_key}_allclasses")
                save_array_and_plot(avg_all, outprefix, title=f"{model_name} {expl_key} allclasses (day {day})")
                model_aggregated[expl_key]['all'] = {'avg': avg_all, 'stack': stack_all, 'n_samples': stack_all.shape[0]}
                print("Saved combined map:", outprefix + ".npy/.png (n={})".format(stack_all.shape[0]))

        # probability-weighted average
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

        # generate probability-weighted maps if stack sizes match
        for expl_key, expl_res in model_aggregated.items():
            if 'all' in expl_res and expl_res['all']['stack'] is not None:
                stack = expl_res['all']['stack']
                if stack.shape[0] == len(pick):
                    weights = pick[:, None]
                    weighted = (stack * weights).sum(axis=0) / (weights.sum() + 1e-9)
                    outprefix = os.path.join(model_out_dir, f"global_{expl_key}_prob_weighted")
                    save_array_and_plot(weighted, outprefix, title=f"{model_name} {expl_key} prob-weighted (day {day})")
                    print("Saved prob-weighted map:", outprefix + ".npy/.png")

        aggregated_maps[model_name] = model_aggregated

    # save a small summary
    with open(os.path.join(out_day_dir, "aggregated_summary.json"), "w") as fh:
        simple = {}
        for mname, exps in aggregated_maps.items():
            simple[mname] = {}
            for ek, cds in exps.items():
                simple[mname][ek] = {ck: {'n_samples': cds[ck].get('n_samples')} for ck in cds.keys()}
        json.dump(simple, fh, indent=2)

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

if __name__ == "__main__":
    main()
