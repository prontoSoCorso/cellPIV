# filename: analyze_temporal_flows.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests

# === Step 1: Merge with Original DB ===
def import_original_db_and_merge_data(data, original_db_path):
    df_db = pd.read_csv(original_db_path)[['dish_well', 'BLASTO NY']]
    merged = pd.merge(data, df_db, on='dish_well', how='left')
    return merged

# === Step 2: Separate and Plot ===
def separate_data(df):
    blasto = df[df['BLASTO NY'] == 1]
    no_blasto = df[df['BLASTO NY'] == 0]
    return blasto, no_blasto

def create_plot(blasto, no_blasto, output_path, temporal_data_type):
    temporal_cols = [col for col in blasto.columns if col.startswith("time_")]
    x = np.arange(len(temporal_cols))

    blasto_mean = blasto[temporal_cols].mean()
    blasto_std = blasto[temporal_cols].std()

    no_blasto_mean = no_blasto[temporal_cols].mean()
    no_blasto_std = no_blasto[temporal_cols].std()

    plt.figure(figsize=(12, 6))
    plt.plot(x, blasto_mean, label="Blasto", color="blue")
    plt.fill_between(x, blasto_mean - blasto_std, blasto_mean + blasto_std, color="blue", alpha=0.2)

    plt.plot(x, no_blasto_mean, label="No Blasto", color="red")
    plt.fill_between(x, no_blasto_mean - no_blasto_std, no_blasto_mean + no_blasto_std, color="red", alpha=0.2)

    plt.title(f"Mean Optical Flow - {temporal_data_type})")
    plt.xlabel("Time step")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)

    plt.savefig(output_path)
    plt.close()

# === Step 3: Statistical Testing ===
def statistical_tests_with_correction(blasto, no_blasto, 
                                      alpha=0.05,
                                      method='fdr_bh',   # bonferroni (più rigido) o fdr_bh 
                                      test_type="mannwhitney"):
    cols = [c for c in blasto.columns if c.startswith("time_")]
    pvals = []
    for c in cols:
        x1 = blasto[c].dropna()
        x2 = no_blasto[c].dropna()
        if test_type=='mannwhitney':
            _, p = mannwhitneyu(x1, x2, alternative='two-sided')
        else:
            _, p = ttest_ind(x1, x2, equal_var=False)
        pvals.append(p)

    # apply multiple test correction
    reject, pvals_corr, _, _ = multipletests(pvals, alpha=alpha, method=method)
    # get contiguous intervals of rejection
    sig_idxs = np.where(reject)[0].tolist()
    intervals = []
    if sig_idxs:
        start = sig_idxs[0]
        for i in range(1,len(sig_idxs)):
            if sig_idxs[i] != sig_idxs[i-1]+1:
                intervals.append((start, sig_idxs[i-1]))
                start = sig_idxs[i]
        intervals.append((start, sig_idxs[-1]))
    return cols, pvals, pvals_corr, reject, intervals

# === Step 4: Plot with shaded significant zones ===
def create_plot_with_significance(blasto, no_blasto, cols, intervals,
                                  title, output_path):
    x = np.arange(len(cols))
    b_mean = blasto[cols].mean()
    b_std  = blasto[cols].std()
    n_mean = no_blasto[cols].mean()
    n_std  = no_blasto[cols].std()

    plt.figure(figsize=(12,6))
    # background shading
    for (s,e) in intervals:
        plt.axvspan(s, e, color='orange', alpha=0.2)

    plt.plot(x, b_mean, label='Blasto',    linewidth=2)
    plt.fill_between(x, b_mean-b_std, b_mean+b_std, alpha=0.2)
    plt.plot(x, n_mean, label='No Blasto', linewidth=2)
    plt.fill_between(x, n_mean-n_std, n_mean+n_std, alpha=0.2)

    plt.title(title)
    plt.xlabel('Time step')
    plt.ylabel('Metric')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# === MAIN WORKFLOW ===
def main():
    temporal_data_path = "/home/phd2/Scrivania/CorsoRepo/cellPIV/_02_temporalData/final_series_csv/mean_magnitude_Farneback.csv"
    original_db_path = "/home/phd2/Scrivania/CorsoRepo/cellPIV/datasets/DB_Morpheus_withID.csv"  # <-- update with real path
    output_folder = "/home/phd2/Scrivania/CorsoRepo/cellPIV/_02_temporalData/final_series_csv/output_plots"
    temporal_data_type = "Farneback"
    os.makedirs(output_folder, exist_ok=True)

    # load & merge
    df_temporal = pd.read_csv(temporal_data_path)
    df = import_original_db_and_merge_data(df_temporal, original_db_path)
    blasto, no_blasto = separate_data(df)

    # Plotting
    plot_path = os.path.join(output_folder, f"plot_mean_{temporal_data_type}.jpg")
    create_plot(blasto, no_blasto, plot_path, temporal_data_type)

    # statistics + correction
    cols, pvals, pvals_corr, reject, intervals = statistical_tests_with_correction(
        blasto, no_blasto,
        alpha=0.05,
        method='bonferroni',         # 'bonferroni' for stricter control
        test_type='mannwhitney'  # or 'ttest'
        )

    # save detailed stats
    stats_df = pd.DataFrame({
        'time_step': cols,
        'p_value':   pvals,
        'p_corr':    pvals_corr,
        'signif':    reject
        })
    stats_df.to_csv(os.path.join(output_folder, "detailed_stats.csv"), index=False)

    # summary of intervals
    with open(os.path.join(output_folder, "summary_intervals.txt"), 'w') as f:
        f.write("Significant intervals (corrected alpha=0.05):\n")
        for s,e in intervals:
            f.write(f"  steps {s} → {e}\n")

    # plot
    plot_path = os.path.join(output_folder, "mean_with_significance.png")
    create_plot_with_significance(
        blasto, no_blasto, cols, intervals,
        title="Mean Flow ± SD with Significant Zones",
        output_path=plot_path
        )

    print("Done. Results in:", output_folder)

if __name__ == "__main__":
    main()
