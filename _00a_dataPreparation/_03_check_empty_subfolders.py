
import os

def check_empty_subfolders(input_dir, log_dir):
    """
    Checks for empty video subfolders (with no valid .jpg images containing '_0_')
    Returns tuple: (total_subfolders, empty_subfolder_paths)
    """
    empty_subfolders = []
    total_subfolders = 0
    
    for year in os.listdir(input_dir):
        year_path = os.path.join(input_dir, year)
        if not os.path.isdir(year_path):
            continue
            
        for video_folder in os.listdir(year_path):
            vf_path = os.path.join(year_path, video_folder)
            if not os.path.isdir(vf_path):
                continue
                
            total_subfolders += 1
            has_valid_image = False
            
            for fname in os.listdir(vf_path):
                if fname.lower().endswith('.jpg') and '_0_' in fname:
                    has_valid_image = True
                    break  # Stop checking this folder once we find one valid image
            
            if not has_valid_image:
                empty_subfolders.append(vf_path)

    print(f"Total video subfolders scanned: {total_subfolders}")
    print(f"Empty subfolders found: {len(empty_subfolders)}")
    print("\nEmpty subfolder paths:")
    for path in empty_subfolders:
        print(f" - {path}")
    
    # Optional: Save results to file
    with open(log_dir, "w") as f:
        f.write(f"Total subfolders: {total_subfolders}\n")
        f.write(f"Empty subfolders: {len(empty_subfolders)}\n\n")
        f.write("Empty folder paths:\n")
        for path in empty_subfolders:
            f.write(f"{path}\n")
        