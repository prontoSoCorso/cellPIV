import os
import shutil
import statistics
import matplotlib.pyplot as plt
import random


def parse_equator_filename(fname):
    """
    Estrae da filename:
      - pdbname (parte iniziale, può contenere 'D')
      - well (int)
      - run (int)
      - time_days (float)
    Supporta pattern con 'D' intermedio nel pdb name.
    Esempio:
      'D2019.04.26_S02266_I0141_D_8_13_0_43581.6681344444.jpg'
    """
    base = os.path.splitext(fname)[0]
    parts = base.split('_')
    run = int(parts[-3])          # run = terzo elemento da destra
    time_days = float(parts[-1])  # tempo in giorni
    well = int(parts[-4])         # well = quarto elemento da destra
    pdbname = "_".join(parts[:-4])
    return pdbname, well, run, time_days


def collect_data(input_dir):
    per_well_times = {}
    per_run_wells = {}
    file_index = []
    # Counters for skipped
    skipped_non_dir_year = 0
    skipped_non_dir_video = 0
    skipped_non_frame = 0
    # Track subfolders -> wells
    subfolder_wells = {}

    for year in os.listdir(input_dir):
        year_path = os.path.join(input_dir, year)
        if not os.path.isdir(year_path):
            skipped_non_dir_year += 1
            continue
        for video_folder in os.listdir(year_path):
            vf_path = os.path.join(year_path, video_folder)
            if not os.path.isdir(vf_path):
                skipped_non_dir_video += 1
                continue
            key_folder = (year, video_folder)
            subfolder_wells[key_folder] = set()
            for fname in os.listdir(vf_path):
                if not fname.lower().endswith('.jpg') or '_0_' not in fname:
                    skipped_non_frame += 1
                    continue
                src = os.path.join(vf_path, fname)
                pdb, well, run, tday = parse_equator_filename(fname)
                per_well_times.setdefault((pdb, well), []).append(tday)
                per_run_wells.setdefault((pdb, run), {})[well] = tday
                file_index.append((src, year, video_folder, pdb, well, run, tday))
    return (per_well_times, per_run_wells, file_index,
            skipped_non_dir_year, skipped_non_dir_video, skipped_non_frame,
            subfolder_wells)
    


def compute_statistics(per_well, per_run):
    initial_wells = len(per_well)
    # Filtra out wells con meno di 250 frame
    per_well_filtered = {k:v for k,v in per_well.items() if len(v) >= 250}
    removed_frame = initial_wells - len(per_well_filtered)

    # Compute deltas per well
    per_well_deltas = {}
    for key, times in per_well_filtered.items():
        ts = sorted(times)
        deltas = [(t2 - t1) * 24 * 60 for t1, t2 in zip(ts, ts[1:])]
        per_well_deltas[key] = deltas

    # Compute avg, min, max, std, outlier count per well
    avg_interval_per_well = {}
    avg_interval_per_well = {k: statistics.mean(v) for k, v in per_well_deltas.items()}
    per_well_stats = {}
    for k, deltas in per_well_deltas.items():
        m = min(deltas)
        mi = deltas.index(m)
        M = max(deltas)
        Mi = deltas.index(M)
        sd = statistics.pstdev(deltas)
        out_count = sum(1 for d in deltas if abs(d - statistics.mean(deltas)) > sd)
        per_well_stats[k] = {'avg': statistics.mean(deltas), 'min': (m, mi), 'max': (M, Mi), 'std': sd, 'out_count': out_count}

    # outlier: avg_int fuori mean ± std
    vals = [s['avg'] for s in per_well_stats.values()]
    mu = statistics.mean(vals)
    sd_glob = statistics.pstdev(vals)
    valid_wells = {k: per_well_filtered[k] 
                   for k, stats in per_well_stats.items()
                   if mu-sd <= stats['avg'] <= mu+sd}
    removed_outlier = len(per_well_filtered) - len(valid_wells)
    outliers = {k: per_well_stats[k]['avg'] 
                for k in per_well_stats 
                if k not in valid_wells}

    # Filtra per run
    # Aggiorna per_run_wells: rimuove riferimenti a well filtrati
    per_run_filtered = {run_key: {w: t for w, t in wells.items() if (run_key[0], w) in valid_wells}
                        for run_key, wells in per_run.items()
                        }

    # 2) conteggi 0-24h e 0-30h
    counts_0_24 = {}
    counts_0_30 = {}
    for key, times in valid_wells.items():
        t0 = min(times)
        rel_h = [(t - t0)*24 for t in times]
        counts_0_24[key] = sum(1 for h in rel_h if h <= 24)
        counts_0_30[key] = sum(1 for h in rel_h if h <= 30)

    def summarize(counts):
        vals = list(counts.values())
        return {'media': statistics.mean(vals), 
                'min': min(vals), 
                'max': max(vals), 
                'mediana': statistics.median(vals)}

    stats_0_24 = summarize(counts_0_24)
    stats_0_30 = summarize(counts_0_30)

    # 3) intervallo medio tra well consecutivi per run
    run_deltas = []
    for wells in per_run_filtered.values():
        ts_sorted = [t for _, t in sorted(wells.items())]
        if len(ts_sorted) < 2: continue
        run_deltas.extend((t2 - t1)*24*60 for t1, t2 in zip(ts_sorted, ts_sorted[1:]))
    avg_interval_between_wells = statistics.mean(run_deltas) if run_deltas else None

    return (initial_wells, removed_frame, removed_outlier,
            valid_wells, avg_interval_per_well, 
            counts_0_24, counts_0_30, stats_0_24, stats_0_30, 
            avg_interval_between_wells, run_deltas,
            outliers, mu, sd_glob, per_well_stats)


def copy_and_rename(file_index, output_base, valid_wells):
    t0_well = {}
    for _, _, _, pdb, well, _, t in file_index:
        key = (pdb, well)
        if key not in valid_wells: continue
        t0_well[key] = min(t0_well.get(key, t), t)

    for src, year, vf, pdb, well, run, t in file_index:
        key = (pdb, well)
        if key not in valid_wells: continue
        rel_min = (t - t0_well[key]) * 24 * 60
        dest_dir = os.path.join(output_base, year, vf)
        os.makedirs(dest_dir, exist_ok=True)
        new = f"{pdb}_{well}_{run}_{rel_min:.1f}min.jpg"
        shutil.copy2(src, os.path.join(dest_dir, new))


def main(input_dir, output_dir, log_file):
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)

    print("----- Collecting data -----")
    (per_well, per_run, files,
     skip_year, skip_video, skip_frame,
     subfolder_wells) = collect_data(input_dir)

    print("----- Computing Stats -----")
    (initial_wells, removed_frame, removed_outlier,
     valid_wells, avg_interval_per_well, 
     counts_0_24, counts_0_30, stats_0_24, stats_0_30, 
     avg_interval_between_wells, run_deltas,
     outliers, mu, sd_glob, per_well_stats) = compute_statistics(per_well, per_run)
    
    num_subfolders = sum(
        1 for year in os.listdir(input_dir)
        for vf in os.listdir(os.path.join(input_dir, year))
        if os.path.isdir(os.path.join(input_dir, year, vf))
        )
    
    print("----- Log writing -----")
    # Scrivi log
    with open(log_file, 'w') as f:
        f.write(f"Numero sottocartelle totali: {num_subfolders}\n")
        f.write(f"Sottocartelle vuote: {num_subfolders-skip_year-skip_video-skip_frame-initial_wells}\n")
        f.write(f"Skipped (not directories - years): {skip_year}\n")
        f.write(f"Skipped (not directories - videos): {skip_video}\n")
        f.write(f"Skipped frames (_0_ filter): {skip_frame}\n")

        f.write(f"Wells unfiltered (unique keys): {initial_wells}\n")
        f.write(f"Removed: {removed_frame} wells (fewer than 250 frames)\n")
        f.write(f"Removed: {removed_outlier} wells (outliers avg interval)\n")

        f.write(f"Valid wells: {len(valid_wells)}\n\n")
        f.write("-- Per-well detailed stats --\n")

        # --- Outlier wells ---
        f.write("Outliers (avg interval):\n")
        for k in sorted(outliers.keys()):
            stats = per_well_stats[k]
            f.write(f"{k}: avg={stats['avg']:.1f}, min={stats['min'][0]:.1f}@{stats['min'][1]}, "
                    f"max={stats['max'][0]:.1f}@{stats['max'][1]}, sd={stats['std']:.1f}, "
                    f"outliers={stats['out_count']}\n")

        f.write("\n\n\nValid wells (inlier) stats:\n")
        for k in sorted(valid_wells.keys()):
            stats = per_well_stats[k]
            f.write(f"{k}: avg={stats['avg']:.1f}, min={stats['min'][0]:.1f}@{stats['min'][1]}, "
                    f"max={stats['max'][0]:.1f}@{stats['max'][1]}, sd={stats['std']:.1f}, "
                    f"outliers={stats['out_count']}\n")

        f.write(f"\nGlobal avg interval: {mu:.1f} ± {sd_glob:.1f} min\n\n")
        f.write("\n-- Aggregate stats 0-24h --\n"); f.write(f"Stats: {stats_0_24}\n")
        f.write("-- Aggregate stats 0-30h --\n"); f.write(f"Stats: {stats_0_30}\n")
        if avg_interval_between_wells: f.write(f"Intervallo medio tra well (min): {avg_interval_between_wells:.1f}\n")

    print("----- Computing histograms -----")
    # Istogramma con conteggi e tick ottimizzati
    def plot_hist(data, title, filename, bin_width=None):
        min_val, max_val = min(data), max(data)
        if not bin_width:
            bin_width = max((max_val - min_val) / 20, 1)
        bins = list(range(int(min_val), int(max_val) + int(bin_width) + 1, int(bin_width)))
        counts, edges, patches = plt.hist(data, bins=bins, edgecolor='black')
        plt.title(title)
        plt.xlabel(title)
        plt.ylabel('Count')
        plt.grid(axis='y', alpha=0.75)
        # Annotate
        for count, edge in zip(counts, edges):
            if count > 0:
                plt.text(edge + bin_width/2, count, int(count), ha='center', va='bottom')
        plt.xticks(edges)
        plt.xlim(min_val - bin_width, max_val + bin_width)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(log_file), filename))
        plt.close()
    plot_hist(list(avg_interval_per_well.values()), 'Avg interval per well (min)', 'hist_avg_interval.png')
    plot_hist(list(counts_0_24.values()), 'Frames 0-24h per well', 'hist_0_24h.png', bin_width=1)
    plot_hist(list(counts_0_30.values()), 'Frames 0-30h per well', 'hist_0_30h.png', bin_width=1)
    if run_deltas:
        plot_hist(run_deltas, 'Interval between wells (min)', 'hist_run_deltas.png')
    print(f"Log and histograms saved in {os.path.dirname(log_file)}")

    print("----- Computing 3 random trend (with std) -----")
    # Esempio andamento tempo per singola well (3 wells casuali)
    sample = random.sample(list(valid_wells.keys()), min(3,len(valid_wells)))
    fig, axs = plt.subplots(2, 3, sharey=True, figsize=(12, 8))
    for i, key in enumerate(sample):
        ts_sorted = sorted(valid_wells[key])
        rel_min = [(t-ts_sorted[0])*24*60 for t in ts_sorted]
        axs[0, i].scatter(range(len(rel_min)), rel_min, s=10)
        axs[0, i].set_title(f"{key} full")
        axs[0, i].grid(True)
        axs[1, i].scatter(range(min(150, len(rel_min))), rel_min[:150], s=10)
        axs[1, i].set_title(f"{key} first 150")
        axs[1, i].grid(True)
        if i == 0:
            axs[0, i].set_ylabel("Minuti")
            axs[1, i].set_ylabel("Minuti")
        for j in [0, 1]:
            axs[j, i].set_xlabel("Frame index")
    fig.tight_layout()
    fig.savefig(os.path.join(log_dir,'well_scatter_subplot.png'))
    plt.close(fig)

    # Plot medio + std per frame across wells
    # costruisci serie per ogni well rel_min list
    all_series = [[(t - min(times)) * 24 for t in sorted(times)]
                  for times in valid_wells.values()]
    max_len = max(len(s) for s in all_series)
    means, stds = [], []
    for i in range(max_len):
        vals = [s[i] for s in all_series if len(s)>i]
        means.append(statistics.mean(vals))
        stds.append(statistics.pstdev(vals))
    x = list(range(len(means)))
    plt.figure()
    plt.errorbar(x, means, yerr=stds, fmt='o', ecolor='grey', capsize=2)
    plt.xlabel("Frame index"); plt.ylabel("Ore dal t0")
    plt.title("Mean ± std per frame across wells")
    plt.grid()
    plt.savefig(os.path.join(log_dir,'mean_std_errorbar.png'))
    plt.close()
