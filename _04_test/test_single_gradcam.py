#!/usr/bin/env python3
"""
test_with_GradCAM_single_video.py

Run Grad-CAM for a single sample (video) from your test split and produce
the per-sample visualizations using the builder's single_channel_output_display.
"""

import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn

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
DAYS = [5]                                      # which day subset to use (1,3,...)
SPECIFIC_VIDEOS = ["D2014.09.25_S1120_I141_5"]  # dish_well id to analyze
MODELS_TO_RUN = ['LSTMFCN', 'ConvTran']      # subset of models to run
OUTPUT_BASE = os.path.join(current_dir, "GRADCAM_single_video_output")
os.makedirs(OUTPUT_BASE, exist_ok=True)

EXPLAINER_TYPES = ["Grad-CAM"]               # list of explainers to request
TARGET_LAYERS_OVERRIDE = {'LSTMFCN': None, 'ConvTran': None}
ACQUISITION_STEP_HOURS = 0.25                # used only for plotting labels (if needed)
# ------------------------------------------------

# ---------- Helpers ----------
def model_has_conv2d(model, layer_name=None):
    import torch.nn as nn
    if layer_name is not None:
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
    a = np.asarray(sample)
    if a.ndim == 0:
        base = np.array([[float(a)]])
    elif a.ndim == 1:
        base = a[np.newaxis, :]
    else:
        time_axis = int(np.argmax(a.shape))
        if time_axis != a.ndim - 1:
            a = np.moveaxis(a, time_axis, -1)
        T = a.shape[-1]
        leading = a.shape[:-1]
        C = 1
        for s in leading:
            C *= int(s)
        base = a.reshape(C, T)
    if use_conv2d:
        base = base.reshape(base.shape[0], 1, base.shape[1])
    return base

def ensure_list_of_samples(samples, use_conv2d):
    return [normalize_sample_for_model(s, use_conv2d=use_conv2d) for s in samples]

# ---------- Main ----------
def run_single_video(day, specific_video, 
                     output_base=OUTPUT_BASE, models_to_run=MODELS_TO_RUN, 
                     use_day_in_result_path=True, use_model_in_result_path=True, create_dir_with_data_name=True):
    
    print(f"\n=== Single video Grad-CAM: day {day} — {specific_video} ===")

    # load test set
    _, _, test_path = conf_train.get_paths(day)
    print("Loading test data from:", test_path)
    df_test = utils.load_data(test_path)
    data_dict = utils.build_data_dict(df_test=df_test)
    _, _, _, _, X_test, y_test = utils._check_data_dict(data=data_dict, require_test=True, only_check_test=True)
    X = utils._sanitize_np(X_test)
    y = np.asarray(y_test).astype(int).ravel()
    names = df_test["dish_well"].tolist()

    if specific_video not in names:
        raise ValueError(f"Specific video '{specific_video}' not found in test set (len test={len(names)}).")

    idx = names.index(specific_video)
    raw_sample = np.asarray(X[idx])         # single sample as numpy array
    data_label = int(y[idx])
    data_name = [specific_video]

    if use_day_in_result_path:
        out_video_dir = os.path.join(output_base, f"day_{day}")
    else:
        out_video_dir = output_base
    os.makedirs(out_video_dir, exist_ok=True)

    for model_name in models_to_run:
        print(f"\n--- Model: {model_name} ---")
        if use_model_in_result_path:
            model_out_dir = os.path.join(out_video_dir, model_name)
        else:
            model_out_dir = out_video_dir
        os.makedirs(model_out_dir, exist_ok=True)

        # load model
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

        # Adapter to normalize unexpected incoming dims
        adapter_model = ModelAdapter(model).to(device)
        adapter_model.eval()

        # save module names for inspection
        named_modules_file = os.path.join(model_out_dir, "named_modules.txt")
        with open(named_modules_file, "w") as fh:
            for n, _ in model.named_modules():
                fh.write(n + "\n")
        print("Wrote named modules to:", named_modules_file)

        # decide conv representation
        override_layer = TARGET_LAYERS_OVERRIDE.get(model_name)
        use_conv2d = model_has_conv2d(model, override_layer) if override_layer else model_has_conv2d(model)

        # special-case: if first Conv2d has in_channels == 1, treat sample as (C,T) because model unsqueezes internally
        if use_conv2d:
            first_conv2d = None
            first_conv_name = None
            for n, m in model.named_modules():
                if isinstance(m, nn.Conv2d):
                    first_conv2d = m
                    first_conv_name = n
                    break
            if first_conv2d is not None and getattr(first_conv2d, "in_channels", None) == 1:
                print(f"Detected Conv2d layer '{first_conv_name}' with in_channels==1; using 2D sample (C,T).")
                use_conv2d = False

        print(f"use_conv2d={use_conv2d} for model {model_name}")

        # normalize single sample into a list for get_cam
        data_list = ensure_list_of_samples([raw_sample], use_conv2d=use_conv2d)   # returns list of processed np arrays
        data_labels = [data_label]
        data_names = data_name

        # choose target layer
        target_layer_name = override_layer
        if target_layer_name is None:
            last = None
            for n, m in model.named_modules():
                if use_conv2d and isinstance(m, torch.nn.Conv2d):
                    last = n
                if (not use_conv2d) and isinstance(m, torch.nn.Conv1d):
                    last = n
            if last is None:
                for n, m in model.named_modules():
                    if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                        last = n
            if last is None:
                raise RuntimeError("Could not find a convolutional layer automatically. Set TARGET_LAYERS_OVERRIDE.")
            target_layer_name = last
        print("Using target layer:", target_layer_name)

        # prepare cam builder (time axis last since our normalize yields time as last dim)
        cam_builder = TorchCamBuilder(model=adapter_model, class_names=["no_blasto", "blasto"], time_axs=-1)

        # final layer identifier inside adapter wrapper: prefix with 'base_model.' as adapter stores model there
        final_layer_name = "base_model." + target_layer_name

        # call get_cam and keep outputs in this video's folder
        try:
            cams_dict, predicted_probs_dict, bar_ranges = cam_builder.get_cam(
                data_list=data_list,
                data_labels=data_labels,
                target_classes=[0, 1],
                display_output=False,    # disable internal display; we use single_channel_output_display later
                explainer_types=EXPLAINER_TYPES,
                target_layers=[final_layer_name],
                softmax_final=False,
                data_names=data_names,
                results_dir_path=model_out_dir,
                aspect_factor=18
            )
        except Exception as e:
            raise RuntimeError(f"cam_builder.get_cam raised an exception: {e}")

        # use built-in display helper to create aggregated per-sample figure(s)
        try:
            cam_builder.single_channel_output_display(
                data_list=data_list,
                data_labels=data_labels,
                predicted_probs_dict=predicted_probs_dict,
                cams_dict=cams_dict,
                channel_names=["Optical Flow Metric - Sum Mean Magnitude"],
                explainer_types=EXPLAINER_TYPES[0],
                target_classes=[0, 1],
                target_layers=[final_layer_name],
                data_names=data_names,
                fig_size=(12, 6),  # Slightly larger to accommodate new elements
                grid_instructions=(1, 1),
                bar_ranges_dict=bar_ranges,
                results_dir_path=model_out_dir,
                dt=ACQUISITION_STEP_HOURS * 3600,  # Convert hours to seconds for compatibility
                line_width=0.6,
                marker_width=40,
                axes_names=("Time (hours)", "Signal"),  # Updated axis label
                create_dir_with_data_name=create_dir_with_data_name      # Avoid extra dir level
            )
            print("single_channel_output_display saved outputs to:", model_out_dir)
        except Exception as e:
            print("single_channel_output_display failed:", e)

    print("All done for single video. Outputs in:", out_video_dir)
    return

def main():
    for DAY in DAYS:
        for SPECIFIC_VIDEO in SPECIFIC_VIDEOS:
            run_single_video(DAY, SPECIFIC_VIDEO)

if __name__ == "__main__":
    main()
