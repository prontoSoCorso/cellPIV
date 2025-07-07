


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



    # Copia e rinomina
    no = False
    if no:
        copy_and_rename(files, scope_final_dir, set(per_well.keys()))
        print(f"Log e plot in: {log_dir}\nCopia/rename fatte.")
