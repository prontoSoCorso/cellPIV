# filename: analyze_temporal_flows.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests

# === Step 1: Merge with Original DB ===
def import_original_db_and_merge_data(data, original_db_path, remove_not_vitals=False):
    df_db = pd.read_csv(original_db_path)[['dish_well', 'BLASTO NY', 'PN']]
    merged = pd.merge(data, df_db, on='dish_well', how='left')
    if remove_not_vitals:
        merged = merged.loc[~merged["PN"].isin(["0PN", "deg"])]
    return merged

# === Step 2: Separate and Plot ===
def separate_data(df):
    blasto = df[df['BLASTO NY'] == 1]
    no_blasto = df[df['BLASTO NY'] == 0]
    return blasto, no_blasto

def create_plot(blasto, no_blasto, output_path, temporal_data_type):
    temporal_cols = [c for c in blasto.columns if c.startswith("time_")]
    x = np.arange(len(temporal_cols))

    b_mean, b_std = blasto[temporal_cols].mean(), blasto[temporal_cols].std()
    n_mean, n_std = no_blasto[temporal_cols].mean(), no_blasto[temporal_cols].std()

    plt.figure(figsize=(12, 6))
    plt.plot(x, b_mean, label="Blasto", color="blue")
    plt.fill_between(x, b_mean - b_std, b_mean + b_std, color="blue", alpha=0.2)
    plt.plot(x, n_mean, label="No Blasto", color="red")
    plt.fill_between(x, n_mean - n_std, n_mean + n_std, color="red", alpha=0.2)

    plt.title(f"Mean Optical Flow – {temporal_data_type}")
    plt.xlabel("Time step"); plt.ylabel("Metric value")
    plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(output_path); plt.close()

# single‐class plot (when only one BLASTO NY present)
def create_plot_single_label(df, output_path, temporal_data_type):
    temporal_cols = [c for c in df.columns if c.startswith("time_")]
    x = np.arange(len(temporal_cols))
    mean, std = df[temporal_cols].mean(), df[temporal_cols].std()

    plt.figure(figsize=(12, 6))
    plt.plot(x, mean, label="Data", color="gray")
    plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.title(f"Mean Optical Flow – {temporal_data_type}")
    plt.xlabel("Time step"); plt.ylabel("Metric value")
    plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(output_path); plt.close()

# === Step 3: Statistical Testing ===
def statistical_tests_with_correction(blasto, no_blasto, time_cols,
                                      alpha=0.05, method='fdr_bh',
                                      test_type='mannwhitney'):
    pvals = []
    for c in time_cols:
        x1, x2 = blasto[c].dropna(), no_blasto[c].dropna()
        if test_type=='mannwhitney':
            _, p = mannwhitneyu(x1, x2, alternative='two-sided')
        else:
            _, p = ttest_ind(x1, x2, equal_var=False)
        pvals.append(p)

    reject, p_corr, _, _ = multipletests(pvals, alpha=alpha, method=method)
    sig_idxs = np.where(reject)[0].tolist()
    intervals = []
    if sig_idxs:
        start = sig_idxs[0]
        for i in range(1, len(sig_idxs)):
            if sig_idxs[i] != sig_idxs[i-1] + 1:
                intervals.append((start, sig_idxs[i-1])); start = sig_idxs[i]
        intervals.append((start, sig_idxs[-1]))
    return time_cols, pvals, p_corr, reject, intervals

# === Step 4: Plot with shaded significant zones ===
def create_plot_with_significance(blasto, no_blasto, cols, intervals,
                                  title, output_path):
    x = np.arange(len(cols))
    b_mean, b_std = blasto[cols].mean(), blasto[cols].std()
    n_mean, n_std = no_blasto[cols].mean(), no_blasto[cols].std()

    plt.figure(figsize=(12, 6))
    for (s, e) in intervals:
        plt.axvspan(s, e, color='orange', alpha=0.2)

    plt.plot(x, b_mean, label='Blasto', linewidth=2)
    plt.fill_between(x, b_mean - b_std, b_mean + b_std, alpha=0.2)
    plt.plot(x, n_mean, label='No Blasto', linewidth=2)
    plt.fill_between(x, n_mean - n_std, n_mean + n_std, alpha=0.2)

    plt.title(title)
    plt.xlabel('Time step'); plt.ylabel('Metric')
    plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close()

# === MAIN WORKFLOW ===
def main():
    # parameters
    data_type          = "sum_mean_mag"
    method_optical_flow= "Farneback"
    start_shift        = 0
    end_frame          = 128
    test_method        = 'mannwhitney'
    correction_method  = 'bonferroni'  # 'bonferroni' / 'fdr_bh'

    # paths
    temporal_data_path = (
        f"/home/phd2/Scrivania/CorsoRepo/cellPIV/"
        f"_02_temporalData/final_series_csv/{data_type}_{method_optical_flow}.csv"
    )
    original_db_path   = (
        "/home/phd2/Scrivania/CorsoRepo/cellPIV/datasets/DB_Morpheus_withID.csv"
    )
    output_root        = os.path.join(
        "/home/phd2/Scrivania/CorsoRepo/cellPIV/"
        "_02_temporalData/final_series_csv/output_plots", data_type
    )
    os.makedirs(output_root, exist_ok=True)

    # load & merge once
    df_temp = pd.read_csv(temporal_data_path)
    df_full = import_original_db_and_merge_data(df_temp, original_db_path, remove_not_vitals=False)

    # A small helper to run the full analysis on any df, under a given subfolder & label:
    def run_analysis(df, subfolder, label_suffix, do_plot_single_if_needed=True):
        out_dir = os.path.join(output_root, subfolder)
        os.makedirs(out_dir, exist_ok=True)

        # if only one class present, just do a single‐line plot
        if do_plot_single_if_needed and df['BLASTO NY'].nunique() < 2:
            create_plot_single_label(
                df,
                os.path.join(out_dir, f"plot_mean_{method_optical_flow}.jpg"),
                method_optical_flow
            )
            return

        blasto, no_blasto = separate_data(df)

        # mean ± SD plot
        create_plot(
            blasto, no_blasto,
            os.path.join(out_dir, f"plot_mean_{method_optical_flow}.jpg"),
            method_optical_flow
        )

        # filter time columns
        time_cols = [c for c in blasto.columns if c.startswith("time_")][start_shift:end_frame]
        bl               = blasto[['dish_well','BLASTO NY'] + time_cols]
        nb               = no_blasto[['dish_well','BLASTO NY'] + time_cols]

        # stats & correction
        cols, pvals, p_corr, reject, intervals = statistical_tests_with_correction(
            bl, nb, time_cols,
            alpha=0.05, method=correction_method, test_type=test_method
        )

        # save detailed stats
        pd.DataFrame({
            'time_step': cols,
            'p_value':   pvals,
            'p_corr':    p_corr,
            'signif':    reject
        }).to_csv(os.path.join(out_dir, "detailed_stats.csv"), index=False)

        # summary intervals
        with open(os.path.join(out_dir, "summary_intervals.txt"), 'w') as f:
            f.write(f"Significant intervals ({label_suffix}, alpha=0.05):\n")
            for s,e in intervals:
                f.write(f"  steps {s} → {e}\n")

        # shaded significance plot
        create_plot_with_significance(
            bl, nb, cols, intervals,
            title=f"{label_suffix} | Mean Flow ± SD with Significant Zones",
            output_path=os.path.join(out_dir, f"mean_with_significance_{correction_method}.png")
        )

    # 1) ALL PN
    run_analysis(df_full, subfolder="all_PN", label_suffix="ALL_PN")

    # 2) ALL PN excluding 0PN/deg
    df_noDeg = df_full.loc[~df_full["PN"].isin(["0PN", "deg"])]
    run_analysis(df_noDeg, subfolder="all_PN_noDeg", label_suffix="ALL_PN_noDeg")

    # 3) per‐PN
    pn_labels = df_full["PN"].dropna().unique()
    for pn in pn_labels:
        run_analysis(
            df_full[df_full["PN"] == pn],
            subfolder=f"PN_{pn}",
            label_suffix=f"PN={pn}"
        )

    print("Done. Results under:", output_root)


if __name__ == "__main__":
    main()
