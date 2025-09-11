#!/usr/bin/env python3
import os
import shutil
import argparse
import csv

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

def load_valid_wells(csv_file):
    """
    Expects CSV with header:
       dish_well,acquisition_hours
    where dish_well is 'pdbname_well'.
    We ignore acquisition_hours here and just return the set of keys.
    """
    valid = set()
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for dish_well, _ in reader:
            # split on last underscore in case pdbname contains underscores
            pdbname, well_str = dish_well.rsplit('_', 1)
            valid.add((pdbname, int(well_str)))
    return valid

def collect_file_index(input_dir):
    """
    Walk input_dir and return a list of tuples:
      (full_path, rel_year, rel_video_folder, pdbname, well, run, time_days)
    Only files ending in .jpg and containing '_0_' are considered.
    """
    file_index = []
    for year in sorted(os.listdir(input_dir)):
        year_path = os.path.join(input_dir, year)
        if not os.path.isdir(year_path):
            continue
        for video in sorted(os.listdir(year_path)):
            vf_path = os.path.join(year_path, video)
            if not os.path.isdir(vf_path):
                continue
            for fname in os.listdir(vf_path):
                if not fname.lower().endswith('.jpg') or '_0_' not in fname:
                    continue
                try:
                    pdb, well, run, tday = parse_equator_filename(fname)
                except Exception:
                    continue
                src = os.path.join(vf_path, fname)
                file_index.append((src, year, video, pdb, well, run, tday))
    return file_index

def copy_and_rename_hpi(input_dir, output_dir, csv_file):
    # load valid wells
    valid_wells = load_valid_wells(csv_file)

    # scan all frames
    file_index = collect_file_index(input_dir)

    # compute t0 for each valid well
    t0 = {}
    for src, year, vf, pdb, well, run, tday in file_index:
        key = (pdb, well)
        if key in valid_wells:
            t0[key] = min(t0.get(key, tday), tday)

    # copy + rename
    for src, year, vf, pdb, well, run, tday in file_index:
        key = (pdb, well)
        if key not in valid_wells:
            continue
        delta_days = tday - t0[key]
        hpi = delta_days * 24
        hpi_str = f"{hpi:.1f}h"
        dest_folder = os.path.join(output_dir, year, vf)
        os.makedirs(dest_folder, exist_ok=True)
        base = os.path.splitext(os.path.basename(src))[0]
        base = base.split('_')[:-1]
        base = "_".join(base)
        new_name = f"{base}_{hpi_str}.jpg"
        shutil.copy2(src, os.path.join(dest_folder, new_name))

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Copy & rename valid equatorial frames with HPI suffix"
    )
    p.add_argument("--input_dir",       required=True,
                   help="Root with year/video subfolders of equator frames")
    p.add_argument("--output_dir",      required=True,
                   help="Where to write HPI‐renamed frames")
    p.add_argument("--csv_file",        required=True,
                   help="CSV of valid wells (pdbname,well,acquisition_hours)")
    args = p.parse_args()

    copy_and_rename_hpi(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        csv_file=args.csv_file
    )
    print("Done copying & renaming valid wells to HPI format.")
