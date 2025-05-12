import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests

# === Helper functions ===

def import_original_db_and_merge_data(data, original_db_path, remove_not_vitals=False):
    df_db = pd.read_csv(original_db_path)[['dish_well', 'BLASTO NY', 'PN']]
    merged = pd.merge(data, df_db, on='dish_well', how='left')
    if remove_not_vitals:
        merged = merged.loc[~merged['PN'].isin(['0PN', 'deg'])]
    return merged

def separate_data(df):
    return df[df['BLASTO NY'] == 1], df[df['BLASTO NY'] == 0]

def get_time_cols(df):
    return [c for c in df.columns if c.startswith('time_')]

def statistical_tests_with_correction(df1, df2, cols, alpha=0.05, method='fdr_bh', test_type='mannwhitney'):
    pvals = []
    for c in cols:
        x1, x2 = df1[c].dropna(), df2[c].dropna()
        if test_type == 'mannwhitney':
            _, p = mannwhitneyu(x1, x2, alternative='two-sided')
        else:
            _, p = ttest_ind(x1, x2, equal_var=False)
        pvals.append(p)
    reject, p_corr, _, _ = multipletests(pvals, alpha=alpha, method=method)
    idxs = np.where(reject)[0]
    intervals = []
    if idxs.size:
        start = idxs[0]
        for i in range(1, len(idxs)):
            if idxs[i] != idxs[i-1] + 1:
                intervals.append((start, idxs[i-1]))
                start = idxs[i]
        intervals.append((start, idxs[-1]))
    return pvals, p_corr, reject, intervals

# Plot stratified by PN (multiple curves) on a given Axes
def plot_stratified(ax, df, title):
    cols = get_time_cols(df)
    x = np.arange(len(cols))
    pn_labels = sorted(df['PN'].dropna().unique())
    for pn in pn_labels:
        sub = df[df['PN'] == pn]
        m, s = sub[cols].mean(), sub[cols].std()
        ax.plot(x, m, label=str(pn))
        ax.fill_between(x, m - s, m + s, alpha=0.1)
    ax.set_title(title, pad=20)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Optical Flow Metric')
    ax.grid(True)
    ax.legend()

# Plot two curves with SD and significance intervals on a given Axes
def plot_with_significance(ax, df_bl, df_nb, cols, intervals, label1, label2):
    x = np.arange(len(cols))
    m1, s1 = df_bl[cols].mean(), df_bl[cols].std()
    m2, s2 = df_nb[cols].mean(), df_nb[cols].std()
    ymax = ax.get_ylim()[1]
    y_offset = ymax * 0.01  # Offset relativo a ymax
    y_pos_start = ymax + y_offset
    y_pos_end = ymax - 5*y_offset
    text_size = 8

    # shade & annotate
    for i, (s_i, e_i) in enumerate(intervals):
        ax.axvspan(s_i, e_i, color='orange', alpha=0.2)
        ax.axvline(s_i, color='lightgrey', linestyle='dotted')
        ax.axvline(e_i, color='lightgrey', linestyle='dotted')
        # alternate up/down offset (the start point is up, end point is down)
        ax.text(s_i, y_pos_start, str(s_i), rotation=0, va='bottom', ha='center', fontsize=text_size)
        ax.text(e_i, y_pos_end, str(e_i), rotation=0, va='bottom', ha='center', fontsize=text_size)

    ax.plot(x, m1, label=label1)
    ax.fill_between(x, m1 - s1, m1 + s1, alpha=0.1)
    ax.plot(x, m2, label=label2)
    ax.fill_between(x, m2 - s2, m2 + s2, alpha=0.1)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Optical Flow Metric')
    ax.grid(True)
    ax.legend()
    ax.set_title(ax.get_title(), pad=20)

# === MAIN WORKFLOW ===

def main():
    # parameters
    data_type           = 'mean_magnitude'    # sum_mean_mag / mean_magnitude
    method_optical_flow = 'Farneback'
    start_shift         = 0
    end_frame           = 128
    test_method         = 'mannwhitney'
    correction_method   = 'fdr_bh'  # fdr_bh / bonferroni

    # paths
    temporal_data_path = (
        f'/home/phd2/Scrivania/CorsoRepo/cellPIV/_02_temporalData/'
        f'final_series_csv/{data_type}_{method_optical_flow}.csv'
    )
    original_db_path   = (
        '/home/phd2/Scrivania/CorsoRepo/cellPIV/datasets/'
        'DB_Morpheus_withID.csv'
    )
    output_root = os.path.join(
        '/home/phd2/Scrivania/CorsoRepo/cellPIV/'
        '_02_temporalData/final_series_csv/output_plots', data_type
    )
    os.makedirs(output_root, exist_ok=True)

    # load & merge
    df_temp = pd.read_csv(temporal_data_path)
    df_full = import_original_db_and_merge_data(df_temp, original_db_path, remove_not_vitals=False)
    df_clean = df_full.loc[~df_full['PN'].isin(['0PN', 'deg'])]
    blasto, no_blasto = separate_data(df_clean)
    cols_all = get_time_cols(df_clean)[start_shift:end_frame]

    # 1) Stratified all_PN: collapse blasto & no_blasto in unico plot a subplots
    different_zoom = [128, 256, 512]
    for max_frame in different_zoom:
        fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        cols_to_consider = get_time_cols(blasto)[:max_frame]
        meta_cols = ["dish_well"] + ["PN"] + ["BLASTO NY"]
        
        plot_stratified(axes1[0], blasto[meta_cols + cols_to_consider], f'Stratified Blasto – {method_optical_flow} (all_PN)')
        plot_stratified(axes1[1], no_blasto[meta_cols + cols_to_consider], f'Stratified No Blasto – {method_optical_flow} (all_PN)')
        # y-limit per stratified_all_PN
        for ax in axes1:
            ax.set_ylim(0, 3.5) if max_frame>150 else ax.set_ylim(0, 2)
        fig1.tight_layout()
        fig1.savefig(os.path.join(output_root, f'stratified_all_PN_{max_frame}frames.png'), dpi=150)
        plt.close(fig1)

    # 2) Pairwise comparison '2PN' vs others, in single figura con subplots per gruppo
    target = '2PN'
    # --- Blasto ---
    others = [pn for pn in sorted(blasto['PN'].dropna().unique()) if pn != target]
    fig2b, axes2b = plt.subplots(1, len(others), figsize=(5*len(others), 5), sharey=True)
    for ax, other in zip(axes2b, others):
        df1 = blasto[blasto['PN'] == target]
        df2 = blasto[blasto['PN'] == other]
        pvals, p_corr, reject, intervals = statistical_tests_with_correction(
            df1[cols_all], df2[cols_all], cols_all, alpha=0.05, method=correction_method, test_type=test_method
        )
        plot_with_significance(ax, df1, df2, cols_all, intervals, f'{target}', f'{other}')
        ax.set_title(f'Blasto: {target} vs {other}', pad=20)
        ax.set_ylim(0, 2)
    fig2b.tight_layout()
    fig2b.savefig(os.path.join(output_root, 'pairwise_2PN_blasto.png'), dpi=150)
    plt.close(fig2b)

    # --- No Blasto ---
    others_nb = [pn for pn in sorted(no_blasto['PN'].dropna().unique()) if pn != target]
    fig2n, axes2n = plt.subplots(1, len(others_nb), figsize=(5*len(others_nb), 5), sharey=True)
    for ax, other in zip(axes2n, others_nb):
        df1 = no_blasto[no_blasto['PN'] == target]
        df2 = no_blasto[no_blasto['PN'] == other]
        pvals, p_corr, reject, intervals = statistical_tests_with_correction(
            df1[cols_all], df2[cols_all], cols_all, alpha=0.05, method=correction_method, test_type=test_method
        )
        plot_with_significance(ax, df1, df2, cols_all, intervals, f'{target}', f'{other}')
        ax.set_title(f'NoBlasto: {target} vs {other}', pad=20)
        ax.set_ylim(0, 2)
    fig2n.tight_layout()
    fig2n.savefig(os.path.join(output_root, 'pairwise_2PN_noblasto.png'), dpi=150)
    plt.close(fig2n)

    # 3) Figura unica: per ogni PN type, subplot con blasto vs no_blasto
    pn_types = sorted(df_clean['PN'].dropna().unique())
    fig3, axes3 = plt.subplots(1, len(pn_types), figsize=(5*len(pn_types), 5), sharey=True)
    for ax, pn in zip(axes3, pn_types):
        df_pn = df_clean[df_clean['PN'] == pn]
        df_b, df_nb = separate_data(df_pn)
        pvals, p_corr, reject, intervals = statistical_tests_with_correction(
            df_b[cols_all], df_nb[cols_all], cols_all, alpha=0.05, method=correction_method, test_type=test_method
        )
        plot_with_significance(ax, df_b, df_nb, cols_all, intervals, 'Blasto', 'No Blasto')
        ax.set_title(f'PN = {pn}', pad=20)
        ax.set_ylim(0, 2)
    fig3.tight_layout()
    fig3.savefig(os.path.join(output_root, 'blasto_vs_noblasto_per_PN.png'), dpi=150)
    plt.close(fig3)

    print('Done. Tutti i plot sono stati salvati in:', output_root)

if __name__ == '__main__':
    main()
