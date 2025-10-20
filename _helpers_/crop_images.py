#!/usr/bin/env python3
"""
Wrapper (hardcoded paths) to run FEMI preprocessing over image folders.

- No argparse: constants are hardcoded below.
- Robust import: tries to locate preprocessing_images.py in likely repo locations.
- Preserves one-level subfolder structure (e.g. blasto/img.png -> outdir/blasto/img.png).
- Sequential processing with safe error handling.
"""

import os
import sys
import time
import importlib
from typing import List


# ---------------------- HARD-CODED CONSTANTS ----------------------
HOURS_THRESHOLD = 24.0

# Edit these if you want different folders (hardcoded by user request)
IMAGES_DIR = f"/home/phd2/Scrivania/CorsoData/blastocisti_images_{HOURS_THRESHOLD}h/"
OUT_DIR = f"/home/phd2/Scrivania/CorsoData/blastocisti_images_{HOURS_THRESHOLD}h_cropped/"
WORKERS = 16  # kept for compatibility though execution is sequential
# ------------------------------------------------------------------

def list_images_with_subdirs(images_dir: str) -> List[str]:
    """
    Return a flat list of relative file paths including first-level subfolder names.
    Example: ['blasto/img1.png', 'no_blasto/img2.png', 'rootfile.png']
    Only descends one level.
    """
    images_dir = os.path.abspath(images_dir.rstrip(os.sep))
    if not os.path.isdir(images_dir):
        return []
    flat = []
    for entry in sorted(os.listdir(images_dir)):
        full = os.path.join(images_dir, entry)
        if os.path.isdir(full):
            for f in sorted(os.listdir(full)):
                f_full = os.path.join(full, f)
                if os.path.isfile(f_full):
                    flat.append(os.path.join(entry, f))
        else:
            if os.path.isfile(full):
                flat.append(entry)
    return flat


def main(images_dir: str, cropped_images_dir: str, workers: int):
    script_dir = os.path.abspath(os.path.dirname(__file__))
    preproc_dir = os.path.abspath(os.path.join(script_dir, "FEMI", "preprocessing"))
    
    # Make module importable
    if preproc_dir not in sys.path:
        sys.path.insert(0, preproc_dir)

    try:
        preproc = importlib.import_module("preprocessing_images")
    except Exception as e:
        print("Failed to import 'preprocessing_images' from:", preproc_dir)
        sys.exit(1)

    # Normalize and ensure trailing slash consistency
    images_dir = os.path.abspath(images_dir)
    cropped_images_dir = os.path.abspath(cropped_images_dir)
    if not images_dir.endswith(os.sep):
        images_dir = images_dir + os.sep
    if not cropped_images_dir.endswith(os.sep):
        cropped_images_dir = cropped_images_dir + os.sep

    # Set module-level variables used by their functions
    preproc.images_dir = images_dir
    preproc.cropped_images_dir = cropped_images_dir

    # Ensure output directory exists
    os.makedirs(cropped_images_dir, exist_ok=True)

    # Gather images (flat list preserving one-level subfolders)
    images = list_images_with_subdirs(images_dir)
    total_images = len(images)

    print(f"Starting preprocessing with {workers} workers (sequential run).")
    print(f"Images dir: {images_dir}")
    print(f"Cropped output dir: {cropped_images_dir}")
    print(f"Total images found: {total_images}")

    if total_images == 0:
        print("No images found in", images_dir)
        return
    
    # Verify process_image exists
    original_process_image = getattr(preproc, "process_image", None)
    if original_process_image is None or not callable(original_process_image):
        print("preprocessing_images.py does not expose a callable 'process_image'. Aborting.")
        sys.exit(1)

    def _process_image_preserve_subdirs(rel_path: str):
        subdir = os.path.dirname(rel_path)
        if subdir:
            out_subdir = os.path.join(cropped_images_dir, subdir)
            os.makedirs(out_subdir, exist_ok=True)
        return original_process_image(rel_path)

    # Sequential processing with simple error handling
    start = time.time()
    results = []
    for image_rel in images:
        try:
            cropped_image = _process_image_preserve_subdirs(image_rel)
            out_path = os.path.join(cropped_images_dir, image_rel)
            cropped_image.save(out_path)
            results.append(cropped_image)
        except Exception as e:
            print(f"[WARN] error processing {image_rel}: {e}")
            continue
    end = time.time()

    print(f"Processed {len(results)}/{total_images} images.")
    print("Time taken:", end-start)
    
    return results


if __name__ == "__main__":
    main(images_dir=IMAGES_DIR, cropped_images_dir=OUT_DIR, workers=WORKERS)


