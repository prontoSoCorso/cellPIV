import os
import shutil
import statistics

def parse_equator_filename(fname):
    """
    Estrae da filename:
      - pdbname
      - well (int)
      - run (int)
      - time_days (float, giorni)
    """
    base = os.path.splitext(fname)[0]
    parts = base.split('_')
    run = int(parts[-3])
    time_days = float(parts[-1])
    well = int(parts[-4])
    pdbname = "_".join(parts[:-4])
    return pdbname, well, run, time_days


def collect_data(input_dir):
    per_well_times = {}
    file_index = []
    for year in os.listdir(input_dir):
        year_path = os.path.join(input_dir, year)
        if not os.path.isdir(year_path):
            continue
        for vf in os.listdir(year_path):
            vf_path = os.path.join(year_path, vf)
            if not os.path.isdir(vf_path):
                continue
            for fname in os.listdir(vf_path):
                if not fname.lower().endswith('.jpg') or '_0_' not in fname:
                    continue
                src = os.path.join(vf_path, fname)
                pdb, well, run, tday = parse_equator_filename(fname)
                key = (pdb, well)
                per_well_times.setdefault(key, []).append(tday)
                file_index.append((src, year, vf, pdb, well, run, tday))
    return per_well_times, file_index


def compute_valid_wells(per_well_times):
    # filtra >250 frame
    filtered = {k: v for k, v in per_well_times.items() if len(v) >= 250}
    # compute avg intervals
    avg_int = {}
    for k, times in filtered.items():
        ts = sorted(times)
        deltas = [(t2 - t1) * 24 * 60 for t1, t2 in zip(ts, ts[1:])]
        avg_int[k] = statistics.mean(deltas)
    # outlier removal mean±std
    vals = list(avg_int.values())
    mu = statistics.mean(vals)
    sd = statistics.pstdev(vals)
    valid = {k for k, a in avg_int.items() if mu - sd <= a <= mu + sd}
    return valid


def copy_and_rename_hpi(file_index, valid_wells, output_dir):
    # trova t0 per well
    t0 = {}
    for src, year, vf, pdb, well, run, tday in file_index:
        key = (pdb, well)
        if key in valid_wells:
            t0[key] = min(t0.get(key, tday), tday)
    # copia e rinomina
    for src, year, vf, pdb, well, run, tday in file_index:
        key = (pdb, well)
        if key not in valid_wells:
            continue
        delta_days = tday - t0[key]
        hpi = delta_days * 24
        hpi_str = f"{hpi:.1f}h"
        dest = os.path.join(output_dir, year, vf)
        os.makedirs(dest, exist_ok=True)
        base = os.path.splitext(os.path.basename(src))[0]
        new_name = f"{base}_{hpi_str}.jpg"
        shutil.copy2(src, os.path.join(dest, new_name))


def main(input_dir, output_dir):
    # 1) raccogli dati
    per_well_times, file_index = collect_data(input_dir)
    # 2) calcola valid wells
    valid_wells = compute_valid_wells(per_well_times)
    print(f"Valid wells: {len(valid_wells)}")
    # 3) copia e rinomina con HPI
    copy_and_rename_hpi(file_index, valid_wells, output_dir)
    print(f"Copied and renamed to {output_dir}")