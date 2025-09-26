#!/usr/bin/env python3
"""
preprocess_and_merge_time_series.py

Pipeline per:
 - caricare pickles (ognuno contiene tipicamente dict: dish_well -> 1D array)
 - caricare file CSV di riferimento (meta) e CSV con acquisition times (dish_well -> 'h1;h2;...')
 - per ogni serie: (opzioni configurabili)
     * smoothing (se metric_name != 'sum_mean_mag')
     * risampling/interpolazione sulla griglia regolare di 15 min (0.25 h)
     * calcolo percentili 1/99 e normalizzazione per serie
     * optional clip in [0,1]
 - salvare i pickle preprocessati (stesso nome file) in OUT_DIR
 - costruire un CSV finale che unisce metadata + colonne temporali (0.00h,0.25h,...)

Configurare i percorsi e le opzioni nella sezione CONFIG.
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd
from collections import OrderedDict

# Aggiungi la directory superiore al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config_02_processTemporalData as config

# ---------------- CONFIG ----------------
# Se usi questo script dentro il tuo progetto, puoi importare la Config_02_processTemporalData
# come facevi prima; per semplicità qui definisco percorsi diretti.

# Pickles
PICKLE_DIR              = config.pickle_dir
PICKLE_LIST             = config.pickle_files

# Path ai CSV
REFERENCE_CSV           = config.reference_file_path
ACQ_TIMES_CSV           = config.acquisition_times_path

# Output
OUT_PICKLE_DIR          = config.out_pickle_dir
FINAL_CSV_PATH          = config.final_csv_path

# Processing options
GRID_STEP_HOURS             = 0.25  # 15 minutes
SMOOTH_METHOD               = 'savgol'  # 'savgol' or 'median' or None
SMOOTH_WINDOW               = 5        # must be odd for savgol
SMOOTH_POLYORDER            = 3
CLIP_AFTER_NORMALIZATION    = True
MIN_POINTS_FOR_1_99         = 100  # fallback to 5-95 if below
TRUNCATE_TO_GRID_EXTENT     = False  # se True tronca i valori fuori grid
TRIMMING_MAX_LENGTH         = 528

os.makedirs(OUT_PICKLE_DIR, exist_ok=True)

# ---------------- helpers ----------------
try:
    from scipy.signal import savgol_filter
    _has_savgol = True
except Exception:
    _has_savgol = False


def load_pickles(pickle_dir, pickle_list=None):
    files = []
    if pickle_list:
        files = [os.path.join(pickle_dir, p) for p in pickle_list]
    else:
        files = sorted([os.path.join(pickle_dir, f) for f in os.listdir(pickle_dir) if f.lower().endswith('.pkl')])
    data = OrderedDict()
    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            with open(fpath, 'rb') as pf:
                obj = pickle.load(pf)
            data[fname] = obj
            print(f"Loaded pickle: {fname} -> {type(obj)}")
        except Exception as e:
            print(f"WARNING: cannot load {fpath}: {e}")
    return data


def load_csv(csv_path):
    return pd.read_csv(csv_path)


def parse_acquisition_times_field(s):
    # s example: "0.0000;0.2515;0.5015;..."
    if pd.isna(s):
        return np.array([])
    parts = [p for p in str(s).split(';') if p.strip()!='']
    try:
        arr = np.array([float(x) for x in parts], dtype=float)
    except Exception:
        # fallback: try comma
        parts = [p for p in str(s).split(',') if p.strip()!='']
        arr = np.array([float(x) for x in parts], dtype=float)
    return arr


def smooth_array(a, method='savgol', window=7, polyorder=2):
    a = np.asarray(a, dtype=float)
    if method is None:
        return a
    if method == 'savgol' and _has_savgol and (window is not None) and (window % 2 == 1) and (len(a) >= window):
        try:
            return savgol_filter(a, window_length=window, polyorder=polyorder)
        except Exception:
            pass
    # fallback median-like simple moving average
    if method in ('median', 'savgol'):
        k = 3
        if len(a) < k:
            return a
        # simple running mean
        kernel = np.ones(k) / k
        # pad
        padded = np.pad(a, (k//2, k//2), mode='edge')
        smooth = np.convolve(padded, kernel, mode='valid')
        return smooth[:len(a)]
    return a


def resample_to_grid(values, original_times, grid_hours):
    """
    values: 1D array (len n)
    original_times: 1D array of times in hours (len m)
    grid_hours: target 1D array of times in hours

    Strategy:
    - if len(original_times) >= len(values): use first len(values) times
    - if len(values) > len(original_times): truncate values to len(original_times)
    - then use np.interp for linear interpolation (extrapolate with NaN outside range)
    """
    values = np.asarray(values, dtype=float)
    times = np.asarray(original_times, dtype=float)
    if len(times) == 0 or len(values) == 0:
        return np.full(len(grid_hours), np.nan)
    if len(times) != len(values):
        # align lengths by truncation
        n = min(len(times), len(values))
        times = times[:n]
        values = values[:n]
    # ensure times are increasing
    order = np.argsort(times)
    times = times[order]
    values = values[order]
    # for np.interp we must define values outside range; we'll set outside to nan
    left = times[0]
    right = times[-1]
    res = np.interp(grid_hours, times, values, left=np.nan, right=np.nan)
    # np.interp with left/right nan won't work; so do manual masking
    mask = (grid_hours >= left) & (grid_hours <= right)
    out = np.full_like(grid_hours, np.nan, dtype=float)
    if mask.any():
        out[mask] = np.interp(grid_hours[mask], times, values)
    return out


def compute_percentile_normalization(arr, p_low=1.0, p_high=99.0, clip=True):
    a = np.asarray(arr, dtype=float)
    if np.all(np.isnan(a)):
        return a, np.nan, np.nan
    p1 = np.nanpercentile(a, p_low)
    p99 = np.nanpercentile(a, p_high)
    denom = (p99 - p1)
    if denom == 0 or np.isnan(denom):
        denom = 1.0
    norm = (a - p1) / denom
    if clip:
        norm = np.clip(norm, 0.0, 1.0)
    return norm, p1, p99

# ---------------- main pipeline ----------------

def main():
    # load inputs
    pickle_data = load_pickles(PICKLE_DIR, PICKLE_LIST)
    if len(pickle_data) == 0:
        print("No pickle files found. Exiting.")
        return
    csv_data = load_csv(REFERENCE_CSV)

    # --- CSV dedup and filter ---
    if 'dish_well' in csv_data.columns:
        csv_data['dish_well'] = csv_data['dish_well'].astype(str)
        before_len = len(csv_data)
        # identify duplicates (keep first)
        dup_mask = csv_data.duplicated(subset='dish_well', keep='first')
        if dup_mask.any():
            print("\n--- Numero Righe Duplicate Trovate (basate su 'dish_well'): ---")
            print(int(dup_mask.sum()))
            print("Lista (prime 20):")
            print(csv_data.loc[dup_mask, 'dish_well'].tolist()[:20])
            print("----------------------------------------------------------\n")
            csv_data = csv_data.drop_duplicates(subset='dish_well', keep='first').copy()
        else:
            print("No duplicate dish_well entries found in reference CSV.")
    else:
        # generic duplicate removal if dish_well not present
        before_len = len(csv_data)
        dup_mask = csv_data.duplicated(keep='first')
        if dup_mask.any():
            csv_data = csv_data.drop_duplicates(keep='first').copy()
            print(f"Dropped {before_len - len(csv_data)} duplicate rows from reference CSV (no 'dish_well' column).")
        else:
            print("No duplicate rows found in reference CSV.")

    # remove rows with PN == '0PN' or 'deg'
    if 'PN' in csv_data.columns:
        pn_series = csv_data['PN'].astype(str).str.strip()
        bad_mask = pn_series.isin(['0PN', 'deg'])
        removed = int(bad_mask.sum())
        if removed > 0:
            removed_list = csv_data.loc[bad_mask, 'dish_well'].astype(str).tolist() if 'dish_well' in csv_data.columns else []
            csv_data = csv_data[~bad_mask].copy()
            print(f"Removed {removed} rows from CSV with PN in ['0PN','deg']. Examples (up to 10):")
            print(removed_list[:10])

    acq_df = load_csv(ACQ_TIMES_CSV)

    # --- TRIMMING OF PICKLES BEFORE BUILDING GRID ---
    print(f"Trimming all series in pickles to max {TRIMMING_MAX_LENGTH} timesteps (before building grid)...")
    trimmed_info = {}
    for pickle_name, obj in list(pickle_data.items()):
        if not isinstance(obj, dict):
            continue
        trimmed_info[pickle_name] = []
        for series_key in list(obj.keys()):
            vals = np.asarray(obj[series_key], dtype=float)
            if vals.size > TRIMMING_MAX_LENGTH:
                # trim
                pickle_data[pickle_name][series_key] = vals[:TRIMMING_MAX_LENGTH]
                trimmed_info[pickle_name].append(series_key)
        if len(trimmed_info[pickle_name]) > 0:
            print(f" {pickle_name}: trimmed {len(trimmed_info[pickle_name])} series to TRIMMING_MAX_LENGTH timesteps")
            for k in trimmed_info[pickle_name][:10]:
                print(" -", k)
            if len(trimmed_info[pickle_name]) > 10:
                print(f" ... and {len(trimmed_info[pickle_name]) - 10} more")

    # build acquisition times dict
    acq_times_dict = {}
    for _, row in acq_df.iterrows():
        key = str(row['dish_well'])
        acq_times_dict[key] = parse_acquisition_times_field(row['acquisition_hours'])

    # build global grid
    all_acq = np.concatenate([v for v in acq_times_dict.values() if v.size>0]) if len(acq_times_dict)>0 else np.array([0.0])
    max_h = float(np.nanmax(all_acq)) if all_acq.size>0 else 0.0
    max_h_limit = TRIMMING_MAX_LENGTH * GRID_STEP_HOURS

    if max_h > max_h_limit:
        print(f"Limiting grid max from {max_h:.2f}h to TRIMMING_MAX_LENGTH*GRID_STEP_HOURS = {max_h_limit:.2f}h")
        max_h = max_h_limit

    # extend a bit to include last
    grid_hours = np.arange(0.0, max_h + GRID_STEP_HOURS + 1e-9, GRID_STEP_HOURS)
    print(f"Grid from 0 to {max_h:.2f} h with step {GRID_STEP_HOURS} -> {len(grid_hours)} columns")

    # storage for processed pickles
    processed_pickles = OrderedDict()

    for pickle_name, obj in pickle_data.items():
        print(f"Processing pickle {pickle_name}...")
        processed_dict = {}
        if not isinstance(obj, dict):
            print(f"  Skipping {pickle_name}: expected dict, got {type(obj)}")
            continue
        # metric_name detection
        metric_name = os.path.splitext(pickle_name)[0]  # e.g. sum_mean_mag
        
        seen_dw = set()
        skipped_duplicates = []
        for series_key, series_values in obj.items():
            # --- EXACT MATCH for acquisition times: use series_key if present in acq_times_dict, else fall back to inferred regular grid
            matched_dw_key = series_key if series_key in acq_times_dict else None

            dedup_id = matched_dw_key if matched_dw_key is not None else series_key
            if dedup_id in seen_dw:
                skipped_duplicates.append((series_key, dedup_id))
                print(f"  Skipping duplicate series {series_key} (maps to {dedup_id})")
                continue
            seen_dw.add(dedup_id)
            
            # Attempt to match dish_well using the series_key to get acquisition times
            acq_times = acq_times_dict.get(matched_dw_key) if matched_dw_key is not None else None
            if acq_times is None:
                # fallback: try to extract a prefix up to last underscore + index
                # if not found, build a regular grid by assuming median spacing 0.25h
                print(f"  Warning: acquisition times not found for {series_key}. Using inferred regular grid.")
                inferred_times = np.arange(0, len(series_values)) * GRID_STEP_HOURS
                acq_times = inferred_times

            # ensure trimming (safety)
            vals = np.asarray(series_values, dtype=float)[:TRIMMING_MAX_LENGTH] #again, only to be sure
            
            # optionally smooth (if not sum_mean_mag)
            if metric_name != 'sum_mean_mag' and SMOOTH_METHOD is not None:
                vals = smooth_array(vals, method=SMOOTH_METHOD, window=SMOOTH_WINDOW, polyorder=SMOOTH_POLYORDER)
            
            # resample to global grid
            resampled = resample_to_grid(vals, acq_times, grid_hours)
            
            # normalize percentiles per series (1-99) (fallback to 5-95 if too few non-nan points)
            valid_points = np.sum(~np.isnan(resampled))
            if valid_points >= MIN_POINTS_FOR_1_99:
                normed, p1, p99 = compute_percentile_normalization(resampled, 1.0, 99.0, clip=CLIP_AFTER_NORMALIZATION)
            else:
                normed, p1, p99 = compute_percentile_normalization(resampled, 5.0, 95.0, clip=CLIP_AFTER_NORMALIZATION)
            
            processed_dict[series_key] = {
                'original_values': np.asarray(series_values, dtype=float),
                'acq_times_used': acq_times,
                'resampled_values': resampled,
                'normalized_values': normed,
                'p1': p1,
                'p99': p99
            }

        # print summary of duplicates skipped for this pickle
        if len(skipped_duplicates) > 0:
            print(f" Skipped {len(skipped_duplicates)} duplicate series in {pickle_name} (kept first occurrence per dish_well).")

        # --- summary matching with csv_data dish_well ---
        csv_dishwells = set(csv_data['dish_well'].astype(str))
        matched_keys = []
        unmatched_keys = []
        # match only exact keys
        for k in processed_dict.keys():
            if k in csv_dishwells:
                matched_keys.append(k)
            else:
                unmatched_keys.append(k)

        print(f"  Processed keys total: {len(processed_dict)}")
        print(f"  Keys matching CSV (will be included in filtered pickle): {len(matched_keys)}")
        print(f"  Keys NOT in CSV (kept only in full pickle): {len(unmatched_keys)}")
        if len(unmatched_keys) > 0:
            print(f"   Example unmatched keys (up to 10):")
            for k in unmatched_keys[:10]:
                print("    -", k)

        # Save: (1) full processed pickle (as before), (2) filtered pickle containing only keys matching CSV
        out_pickle_path_full = os.path.join(OUT_PICKLE_DIR, pickle_name)
        to_save_full = {k: v['normalized_values'] for k, v in processed_dict.items()}
        with open(out_pickle_path_full, 'wb') as pf:
            pickle.dump(to_save_full, pf)

        # filtered pickle name: add suffix "_filtered_by_csv.pkl"
        base, ext = os.path.splitext(pickle_name)
        out_pickle_path_filtered = os.path.join(OUT_PICKLE_DIR, f"{base}_filtered_by_csv{ext}")
        to_save_filtered = {k: processed_dict[k]['normalized_values'] for k in matched_keys}
        with open(out_pickle_path_filtered, 'wb') as pf:
            pickle.dump(to_save_filtered, pf)

        print(f"  Saved full processed pickle to {out_pickle_path_full} (n_entries={len(to_save_full)})")
        print(f"  Saved filtered processed pickle to {out_pickle_path_filtered} (n_entries={len(to_save_filtered)})")

        # keep the processed_dict in memory as before
        processed_pickles[pickle_name] = processed_dict


    # ---------------- build merged CSV ----------------
    # choose which processed metric to use for time-series columns: prefer sum_mean_mag if present
    chosen_metric_name = None
    if 'sum_mean_mag.pkl' in processed_pickles:
        chosen_metric_name = 'sum_mean_mag.pkl'
    else:
        # take first
        chosen_metric_name = next(iter(processed_pickles.keys())) if len(processed_pickles)>0 else None

    if chosen_metric_name is None:
        print("No processed metric available to build FinalDataset.csv. Exiting.")
        return

    print(f"Building FinalDataset using metric: {chosen_metric_name}")

    proc_for_csv = processed_pickles[chosen_metric_name]

    # prepare grid column names like '0.00h','0.25h'...
    grid_col_names = [f"{h:.2f}h" for h in grid_hours]

    rows = []
    for _, meta_row in csv_data.iterrows():
        dish_well = str(meta_row['dish_well'])
        patient_id = meta_row.get('patient_id', '')
        # metadata fields requested
        blasto = meta_row.get('BLASTO NY', '')
        eup = meta_row.get('eup_aneup', '')
        pn = meta_row.get('PN', '')
        maternal_age = meta_row.get('maternal age', '')

        # --- EXACT MATCH only ---
        matched_key = dish_well if dish_well in proc_for_csv else None

        if matched_key is None:
            series_vals = [np.nan] * len(grid_hours)
        else:
            series_vals = proc_for_csv[matched_key]['normalized_values']
            # ensure same length as grid
            if len(series_vals) != len(grid_hours):
                # try to resample again using stored acq times
                acq_times_used = proc_for_csv[matched_key]['acq_times_used']
                series_vals = resample_to_grid(proc_for_csv[matched_key]['original_values'], acq_times_used, grid_hours)
                # normalize again
                series_vals, _, _ = compute_percentile_normalization(series_vals, 1.0, 99.0, clip=CLIP_AFTER_NORMALIZATION)
                # if still mismatch, pad/truncate
                if len(series_vals) != len(grid_hours):
                    N = len(grid_hours)
                    tmp = np.full(N, np.nan)
                    tmp[:min(N, len(series_vals))] = series_vals[:min(N, len(series_vals))]
                    series_vals = tmp
        row = {
            'patient_id': patient_id,
            'dish_well': dish_well,
            'BLASTO NY': blasto,
            'eup_aneup': eup,
            'PN': pn,
            'maternal age': maternal_age
        }
        # append time columns
        for cname, val in zip(grid_col_names, series_vals):
            row[cname] = float(val) if (val is not None and (not np.isnan(val))) else np.nan
        rows.append(row)

    final_df = pd.DataFrame(rows)
    final_df.to_csv(FINAL_CSV_PATH, index=False)
    print(f"Saved FinalDataset CSV to {FINAL_CSV_PATH} (n_rows={len(final_df)})")

    # Diagnostic summary after final CSV
    csv_dishwells = set(csv_data['dish_well'].astype(str))
    pk_keys = set()
    for pdict in processed_pickles.values():
        pk_keys.update(pdict.keys())
    pk_exact_matches = sum(1 for k in pk_keys if k in csv_dishwells)
    csv_with_series = sum(1 for d in csv_dishwells if d in pk_keys)
    print("Diagnostic final summary:")
    print(f"  CSV rows (after filter): {len(csv_dishwells)}")
    print(f"  PK keys total processed: {len(pk_keys)}")
    print(f"  PK keys exactly equal to a CSV dish_well: {pk_exact_matches}")
    print(f"  CSV dish_well that have an exact PK key: {csv_with_series}")

    print("All done.")

if __name__ == '__main__':
    main()
