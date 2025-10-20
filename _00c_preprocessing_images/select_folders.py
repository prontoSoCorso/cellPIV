import os
import shutil
import pandas as pd

# Paths
csv_path = "/home/phd2/Scrivania/CorsoRepo/cellPIV/datasets/Farneback/FinalDataset.csv"
source_base = "/home/phd2/Scrivania/CorsoData/blastocisti"
target_base = "/home/phd2/Scrivania/CorsoData/blastocisti_filtered"

# Read CSV
df = pd.read_csv(csv_path)

# Loop through each row
for _, row in df.iterrows():
    dish_well = row['dish_well']
    # Find which subfolder contains the dish_well folder
    found = False
    for subfolder in ['blasto', 'no_blasto']:
        src_folder = os.path.join(source_base, subfolder, dish_well)
        if os.path.exists(src_folder):
            dst_folder = os.path.join(target_base, subfolder, dish_well)
            os.makedirs(os.path.dirname(dst_folder), exist_ok=True)
            shutil.copytree(src_folder, dst_folder)
            print(f"Copied {src_folder} to {dst_folder}")
            found = True
            break
    if not found:
        print(f"Folder for {dish_well} not found in source directories.")