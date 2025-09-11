import os
import shutil
import statistics
import matplotlib.pyplot as plt
import random
import numpy as np
from matplotlib.ticker import FuncFormatter


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
    


def compute_statistics(per_well, per_run, min_frames=250, max_gap_minutes=90, ideal_frame_gap=15, hours_to_consider=30):
    """
    Compute per-well and per-run statistics.
    Exclude wells with fewer than `min_frames` or with any gap > `max_gap_minutes` within first 30h.
    Returns also a set of `corrupted_wells` that were removed due to large gaps.
    """
    initial_wells = len(per_well)
    # Filter wells by frame count
    per_well_filtered = {k:v for k,v in per_well.items() if len(v) >= min_frames}
    removed_wells = initial_wells - len(per_well_filtered)

    # Compute deltas per well
    per_well_deltas = {}
    per_well_stats = {}
    # Process each well and drop if blackout gap detected
    corrupted_wells = set()
    corrupted_stats = {}
    for key, times in list(per_well_filtered.items()):
        ts = sorted(times)
        t0 = ts[0]
        
        # cutoff based on ideal number of frame for the first n hours
        num_ideal_frames = int(hours_to_consider*(np.ceil(60/ideal_frame_gap)))
        ts_frames = ts[:num_ideal_frames]

        # compute deltas in minutes
        raw_deltas = [(t2 - t1) * 24 * 60 for t1, t2 in zip(ts_frames, ts_frames[1:])]
        # identify blackout gap --> it must not have significant gap in the first n frames (based on ideal acquisition gap)
        for idx, gap in enumerate(raw_deltas):
            if gap > max_gap_minutes:
                corrupted_wells.add(key)
                corrupted_stats[key] = statistics.mean(raw_deltas)
                per_well_filtered.pop(key)
                break
        if key in corrupted_wells:
            continue

        # now deltas and statistics on clean segment
        # compute stats on full 30h
        cutoff = t0 + (hours_to_consider/24)
        ts30 = [t for t in ts if t <= cutoff]
        deltas = [(t2 - t1) * 24 * 60 for t1, t2 in zip(ts30, ts30[1:])]
        per_well_deltas[key] = deltas
        avg = statistics.mean(deltas)
        m, mi = min(deltas), deltas.index(min(deltas))
        M, Mi = max(deltas), deltas.index(max(deltas))
        sd = statistics.pstdev(deltas)
        per_well_stats[key] = {
            'avg': avg,
            'min': (m, mi),
            'max': (M, Mi),
            'std': sd,
            'frame_count_30h': len(ts30)
        }

    # globals
    vals = [s['avg'] for s in per_well_stats.values()]
    mu, sd_glob = statistics.mean(vals), statistics.pstdev(vals)

    # outlier detection
    for key, times in per_well_filtered.items():
        deltas = per_well_deltas[key]
        avg = statistics.mean(deltas)
        out_count = sum(1 for d in deltas if abs(d - avg) > 2*sd_glob)
        per_well_stats[key]['out_count'] = out_count

    outliers = {}
    expected_frames_for_well = np.ceil(hours_to_consider * 60/mu)
    tolerance_perc = 0.05 # percentuale missing values tollerabile
    tolerance = int(np.ceil(tolerance_perc*expected_frames_for_well))
    for key, stats in per_well_stats.items():
        is_avg_outlier = stats['avg'] >= mu + 2*sd_glob
        is_frame_count_outlier = int(expected_frames_for_well - stats['frame_count_30h']) > tolerance
        if is_avg_outlier and is_frame_count_outlier:
            outliers[key] = stats['avg']

    # Build valid wells (inliers)
    valid_wells = {k: per_well_filtered[k] for k in per_well_stats if k not in outliers}
    removed_outlier = len(per_well_filtered) - len(valid_wells)

    # Filtra per run
    # Aggiorna per_run_wells: rimuove riferimenti a well filtrati
    per_run_filtered = {run_key: {w: t for w, t in wells.items() if (run_key[0], w) in valid_wells}
                        for run_key, wells in per_run.items()
                        }

    # counts
    def get_counts(keys):
        c24, c30 = {}, {}
        for k in keys:
            ts = sorted(per_well[k])
            t0 = ts[0]
            rel = [(t - t0)*24 for t in ts if t <= t0 + hours_to_consider/24]
            c24[k] = sum(h <= 24 for h in rel)
            c30[k] = len(rel)
        return c24, c30
    

    # compute counts for all three groups and entire initial set
    all_keys = set(per_well.keys())
    c24_all, c30_all = get_counts(all_keys)
    c24_valid, c30_valid = get_counts(valid_wells)
    c24_out, c30_out     = get_counts(outliers)
    c24_corr, c30_corr   = get_counts(corrupted_wells)


    def summarize(counts):
        if not counts: return {}
        vals = list(counts.values())
        return {'media': statistics.mean(vals), 
                'min': min(vals), 
                'max': max(vals), 
                'mediana': statistics.median(vals)}

    stats_24h_all = summarize(c24_all)
    stats_30h_all = summarize(c30_all)
    stats_24h_valid = summarize(c24_valid)
    stats_30h_valid = summarize(c30_valid)
    stats_24h_out = summarize(c24_out)
    stats_30h_out = summarize(c30_out)
    stats_24h_corr = summarize(c24_corr)
    stats_30h_corr = summarize(c30_corr)

    # 3) intervallo medio tra well consecutivi per run
    run_deltas = []
    for wells in per_run_filtered.values():
        ts_sorted = [t for _, t in sorted(wells.items())]
        if len(ts_sorted) < 2: continue
        run_deltas.extend((t2 - t1)*24*60 for t1, t2 in zip(ts_sorted, ts_sorted[1:]))
    avg_interval_between_wells = statistics.mean(run_deltas) if run_deltas else None

    return (initial_wells, removed_wells, removed_outlier, corrupted_wells, outliers,
            valid_wells, mu, sd_glob, 
            c24_all, c30_all, stats_24h_all, stats_30h_all,
            c24_valid, c30_valid, stats_24h_valid, stats_30h_valid,
            c24_out, c30_out, stats_24h_out, stats_30h_out, 
            c24_corr, c30_corr, stats_24h_corr, stats_30h_corr, corrupted_stats, 
            avg_interval_between_wells, run_deltas, per_well_stats)


# Istogramma con conteggi e tick ottimizzati
def plot_hist(data, title, filename, bin_width=None, bin_count=15):
    if not data: 
        print("Zero outliers well")
        return
    min_val, max_val = min(data), max(data)
    if bin_width is not None:
        # crea bins ad intervalli fissi
        bins = np.arange(min_val, max_val + bin_width, bin_width)
    else:
        # crea un numero di bins per maggiore risoluzione visiva
        bins = bin_count
    counts, edges, patches = plt.hist(data, bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel(title)
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.75)
    # Annotate
    for count, edge in zip(counts, edges):
        if count > 0:
            # posizione centrale del bin
            mid = edge + ((edges[1]-edges[0]) if isinstance(edges, np.ndarray) else (max_val-min_val)/len(counts))/2
            plt.text(mid, count, int(count), ha='center', va='bottom')
    if isinstance(edges, np.ndarray):
        # reduce xticks to max 10 for readability
        max_ticks = 15
        step = max(1, len(edges)//max_ticks)
        plt.xticks(edges[::step])
        plt.xticks(rotation=45, ha='right') # Ruota di 45 gradi e allinea a destra
        # --- Inizio formattazione dinamica ---
        
        # Determina il range e la grandezza media dei bin
        # Controlla che 'edges' abbia almeno due elementi per evitare IndexError
        if len(edges) > 1:
            bin_size = edges[1] - edges[0]
        else:
            bin_size = 0 # O gestisci come errore/caso limite

        # Funzione personalizzata per la formattazione dei tick
        def custom_formatter(x, pos):
            # Condizione 1: Valore intero (o molto vicino a un intero)
            # Usiamo un piccolo epsilon per tollerare errori di floating-point
            if abs(x - round(x)) < 1e-9: # Se la differenza dall'intero più vicino è minima
                return f"{int(x)}"
            
            # Condizione 2: La distanza tra i bin è molto piccola
            # Qui il tuo criterio: se la distanza tra bin è < 0.1 (o un altro valore che decidi tu)
            # mostriamo 2 cifre decimali. Puoi regolare il 0.1.
            elif bin_size > 0 and bin_size < 0.5: # Se il passo è piccolo, usa più precisione
                 return f"{x:.2f}" # Due cifre decimali
            
            # Condizione 3: Tutti gli altri casi (una cifra decimale)
            else:
                return f"{x:.1f}"

        # Applica il formattatore personalizzato
        plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_formatter))
        
    else:
        # for integer bins use default
        pass

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main(input_dir, output_file, log_file):
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)

    print("----- Collecting data -----")
    (per_well, per_run, files,
     skip_year, skip_video, skip_frame,
     subfolder_wells) = collect_data(input_dir)

    print("----- Computing Stats -----")
    min_frames=250
    max_gap_minutes=61
    ideal_frame_gap=15
    hours_to_consider=30
    (initial_wells, removed_wells, removed_outlier, corrupted_wells, outliers,
     valid_wells, mu, sd_glob, 
     c24_all, c30_all, stats_24h_all, stats_30h_all,
     c24_valid, c30_valid, stats_24h_valid, stats_30h_valid,
     c24_out, c30_out, stats_24h_out, stats_30h_out, 
     c24_corr, c30_corr, stats_24h_corr, stats_30h_corr, corrupted_stats,
     avg_interval_between_wells, run_deltas, per_well_stats) = compute_statistics(per_well, 
                                                                                  per_run, 
                                                                                  min_frames=min_frames, 
                                                                                  max_gap_minutes=max_gap_minutes, 
                                                                                  ideal_frame_gap=ideal_frame_gap,
                                                                                  hours_to_consider=hours_to_consider)
    
    num_subfolders = sum(
        1 for year in os.listdir(input_dir)
        for vf in os.listdir(os.path.join(input_dir, year))
        if os.path.isdir(os.path.join(input_dir, year, vf))
        )
    
    # --- Save valid wells acquisition times (in hours relative to first frame) ---
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as vf:
        vf.write('dish_well,acquisition_hours\n') # Modificato il nome delle colonne
        for (pdb, well), times in sorted(valid_wells.items()):
            ts_sorted = sorted(times)
            t0 = ts_sorted[0]
            hours = [(t - t0) * 24 for t in ts_sorted]
            hours_str = ';'.join(f"{h:.4f}" for h in hours)
            vf.write(f"{pdb}_{well},{hours_str}\n") # Combinato pdb e well per dish_well

    print(f"Valid wells acquisition times saved to {output_file}")

    # --- Scrittura log ---
    print("----- Log writing -----")
    with open(log_file, 'w') as f:
        # --- General stats ---
        f.write(f"Numero sottocartelle totali: {num_subfolders}\n")
        f.write(f"Sottocartelle vuote: {num_subfolders-skip_year-skip_video-skip_frame-initial_wells}\n")
        f.write(f"Skipped (not directories - years): {skip_year}\n")
        f.write(f"Skipped (not directories - videos): {skip_video}\n")
        f.write(f"Skipped frames (_0_ filter): {skip_frame}\n")

        f.write(f"Wells unfiltered (unique keys): {initial_wells}\n")
        f.write(f"Removed: {removed_wells} wells (fewer than 250 frames)\n")
        f.write(f"Removed: {removed_outlier} wells (outliers avg interval)\n")
        f.write(f"Valid wells: {len(valid_wells)}\n\n")

        # --- Detailed per-wells stats ---
        f.write("-- Detailed per-well stats (first 30h) --\n")
        
        # Corrupted wells ---
        f.write(f"\n\nFound {len(corrupted_wells)} Corrupted wells (large gap > {max_gap_minutes} min) in the firsts 30h:\n")
        for k in sorted(corrupted_wells):
            f.write(f"{k}\n")

        # Outlier wells ---
        f.write(f"\n\nOutliers (avg interval), n={removed_outlier}:\n")
        for k in sorted(outliers.keys()):
            stats = per_well_stats[k]
            f.write(f"{k}: avg={stats['avg']:.1f}, min={stats['min'][0]:.1f}@{stats['min'][1]}, "
                    f"max={stats['max'][0]:.1f}@{stats['max'][1]}, sd={stats['std']:.1f}, "
                    f"outliers={stats['out_count']}\n")

        # Valid wells ---
        f.write(f"\n\n\nValid wells (inlier) stats, n={len(valid_wells)}:\n")
        for k in sorted(valid_wells.keys()):
            stats = per_well_stats[k]
            f.write(f"{k}: avg={stats['avg']:.1f}, min={stats['min'][0]:.1f}@{stats['min'][1]}, "
                    f"max={stats['max'][0]:.1f}@{stats['max'][1]}, sd={stats['std']:.1f}, "
                    f"outliers={stats['out_count']}\n")

        f.write(f"\nGlobal avg interval: {mu:.1f} ± {sd_glob:.1f} min\n\n")
        f.write(f"ALL 0-24h stats: {stats_24h_all}\n")
        f.write(f"Valid 0-24h stats: {stats_24h_valid}\n")
        f.write(f"Outliers 0-24h stats: {stats_24h_out}\n")
        f.write(f"Corrupted 0-24h stats: {stats_24h_corr}\n")

        f.write(f"ALL 0-24h stats: {stats_30h_all}\n")
        f.write(f"Valid 0-24h stats: {stats_30h_valid}\n")
        f.write(f"Outliers 0-24h stats: {stats_30h_out}\n")
        f.write(f"Corrupted 0-24h stats: {stats_30h_corr}\n")
        
        if avg_interval_between_wells: f.write(f"Intervallo medio tra well (min): {avg_interval_between_wells:.1f}\n")


    # Histograms
    print("----- Computing histograms -----")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Mean intervals
    n_ideal_frames = int(hours_to_consider*(np.ceil(60/ideal_frame_gap)))
    plot_hist([per_well_stats[k]['avg'] for k in valid_wells],  'Avg delta Valid',                          os.path.join(os.path.dirname(log_file), 'hist_avg_interval_valid.png'))
    plot_hist([per_well_stats[k]['avg'] for k in outliers],     'Avg delta Outliers',                       os.path.join(os.path.dirname(log_file), 'hist_avg_interval_outliers.png'))
    plot_hist([corrupted_stats[k] for k in corrupted_wells],    f'Avg delta Corrupted for {n_ideal_frames}',os.path.join(os.path.dirname(log_file), 'hist_avg_interval_corrupted.png'))

    plot_hist(list(c24_all.values()),   'Frames 0-24h All',       os.path.join(os.path.dirname(log_file),'hist_24_all.png'), bin_width=1)
    plot_hist(list(c30_all.values()),   'Frames 0-30h All',       os.path.join(os.path.dirname(log_file),'hist_30_all.png'), bin_width=1)
    plot_hist(list(c24_valid.values()), 'Frames 0-24h Valid',     os.path.join(os.path.dirname(log_file),'hist_24_valid.png'), bin_width=1)
    plot_hist(list(c30_valid.values()), 'Frames 0-30h Valid',     os.path.join(os.path.dirname(log_file),'hist_30_valid.png'), bin_width=1)
    plot_hist(list(c24_out.values()),   'Frames 0-24h Outliers',  os.path.join(os.path.dirname(log_file),'hist_24_outliers.png'), bin_width=1)
    plot_hist(list(c30_out.values()),   'Frames 0-30h Outliers',  os.path.join(os.path.dirname(log_file),'hist_30_outliers.png'), bin_width=1)
    plot_hist(list(c24_corr.values()),  'Frames 0-24h Corrupted', os.path.join(os.path.dirname(log_file),'hist_24_corrupted.png'), bin_width=1)
    plot_hist(list(c30_corr.values()),  'Frames 0-30h Corrupted', os.path.join(os.path.dirname(log_file),'hist_30_corrupted.png'), bin_width=1)
    
    if run_deltas:
        plot_hist(run_deltas, 
                  'Interval between wells (min)', 
                  os.path.join(os.path.dirname(log_file), 'hist_run_deltas.png'))
    print(f"Log and histograms saved in {os.path.dirname(log_file)}")


if __name__ == '__main__':
    print("===== ATTENZIONE: FARE IL RUN DEL FILE DA _mainDataPreparation =====")