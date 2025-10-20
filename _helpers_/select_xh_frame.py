from __future__ import annotations
import argparse
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Iterable, Dict
import pandas as pd
import sys
import hashlib
import logging

#!/usr/bin/env python3
"""
Utility to extract, for every video folder, the first image whose encoded hour
is >= a selected threshold (e.g., 24.0h) from the blastocyst dataset structure:

    /home/phd2/Scrivania/CorsoData/blastocisti/
            blasto/<video_id>/*.png|jpg|tif...
            no_blasto/<video_id>/*...

It copies the selected frame into an output structure:

    /home/phd2/Scrivania/CorsoData/blastocisti_images_{H}h/
            blasto/original_images/
            no_blasto/original_images/
    plus a manifest CSV with metadata.

Filename hour parsing uses a regex searching for '<number>h' (case-insensitive),
optionally with decimals, e.g.: 23h, 24.0h, _24.5h, frame_024.0h.png

Usage (example):
    python select_xh_frame.py --hours 24.0

Optional:
    --base /custom/base/path
    --out-root /custom/output/root
    --dry-run
    --copy-mode copy|symlink|hardlink

Produces:
    manifest_24.0h.csv inside the output root.

"""


# Configurable constants
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
HOUR_REGEX = re.compile(r'(?i)(?:^|[_\-\s])(\d+(?:\.\d+)?)h(?:_|\.|$)')

CLASS_MAP = [
        ("blasto", 1),
        ("no_blasto", 0),
]

DEFAULT_BASE = Path("/home/phd2/Scrivania/CorsoData/blastocisti")
DEFAULT_OUTPUT_PARENT = Path("/home/phd2/Scrivania/CorsoData")
HOURS_THRESHOLD = 24.0
OUT_ROOT = Path(f"/home/phd2/Scrivania/CorsoData/blastocisti_images_{HOURS_THRESHOLD}h")
COPY_MODE = "copy"          # copy | symlink | hardlink
DRY_RUN = False
NO_MANIFEST = False

logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s"
)

@dataclass
class SelectedImage:
        video_id: str
        class_name: str
        y_true: int
        hour: float
        source_path: Path
        dest_path: Path


def parse_hour_from_name(name: str) -> Optional[float]:
        """
        Extract hour value from filename based on pattern '<number>h'.
        Returns None if not found or not convertible.
        """
        m = HOUR_REGEX.search(os.path.basename(name))
        if not m:
                return None
        try:
                return float(m.group(1))
        except Exception:
                return None


def list_images(folder: Path) -> List[Path]:
        return [
                p for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in IMG_EXT
        ]


def first_image_at_or_after_xh(video_dir: Path, hours_threshold: float) -> Optional[Tuple[Path, float]]:
        """
        Returns (image_path, hour) of the first image whose hour >= threshold
        based on ascending hour order. None if not found.
        """
        candidates: List[Tuple[Path, float]] = []
        for img in list_images(video_dir):
                h = parse_hour_from_name(img.name)
                if h is not None and h >= hours_threshold:
                        candidates.append((img, h))
        if not candidates:
                return None
        candidates.sort(key=lambda x: x[1])
        return candidates[0]


def safe_copy(src: Path, dst: Path, mode: str = "copy", dry_run: bool = False):
        """
        Copy/link file according to mode: copy | symlink | hardlink
        """
        if dst.exists():
                return
        if dry_run:
                logging.info(f"[DRY] {mode} -> {dst}")
                return
        dst.parent.mkdir(parents=True, exist_ok=True)
        if mode == "copy":
                shutil.copy2(src, dst)
        elif mode == "symlink":
                os.symlink(src, dst)
        elif mode == "hardlink":
                os.link(src, dst)
        else:
                raise ValueError(f"Unknown copy mode: {mode}")


def hash_short(path: Path) -> str:
        h = hashlib.sha1(str(path).encode()).hexdigest()
        return h[:8]


def plan_selection(
        base_dir: Path,
        hours_threshold: float,
        out_root: Path,
        copy_mode: str = "copy"
) -> List[SelectedImage]:
        """
        Scan dataset and plan selected frames.
        """
        selections: List[SelectedImage] = []
        for class_name, y_true in CLASS_MAP:
                class_dir = base_dir / class_name
                if not class_dir.exists():
                        logging.warning(f"Missing class dir: {class_dir}")
                        continue
                video_dirs = sorted([p for p in class_dir.iterdir() if p.is_dir()])
                for video_dir in video_dirs:
                        sel = first_image_at_or_after_xh(video_dir, hours_threshold)
                        if sel is None:
                                continue
                        img_path, hour = sel
                        # Destination directory pattern: out_root/<class_name>
                        dest_dir = out_root / class_name
                        # To avoid collisions across videos using identical filenames,
                        # prefix with video_id or add hash.
                        dest_name = f"{img_path.name}"
                        # If still length issues or duplicates, add hash.
                        dest_path = dest_dir / dest_name
                        if dest_path.exists():
                                # Add short hash suffix to disambiguate
                                stem, ext = os.path.splitext(dest_name)
                                dest_path = dest_dir / f"{stem}_{hash_short(img_path)}{ext}"
                        selections.append(SelectedImage(
                                video_id=video_dir.name,
                                class_name=class_name,
                                y_true=y_true,
                                hour=hour,
                                source_path=img_path,
                                dest_path=dest_path
                        ))
        return selections


def execute_selection(
        selections: List[SelectedImage],
        copy_mode: str,
        dry_run: bool
) -> None:
        for s in selections:
                safe_copy(s.source_path, s.dest_path, mode=copy_mode, dry_run=dry_run)


def main() -> int:

        if not DEFAULT_BASE.exists():
                logging.error(f"Base directory not found: {DEFAULT_BASE}")
                return 2

        if OUT_ROOT is None:
                logging.error("Output root not specified")
                return 3

        logging.info(f"Hour threshold: {HOURS_THRESHOLD}")
        logging.info(f"Base dir: {DEFAULT_BASE}")
        logging.info(f"Output root: {OUT_ROOT}")
        logging.info(f"Copy mode: {COPY_MODE}")
        if DRY_RUN:
                logging.info("DRY RUN enabled (no file copies will occur)")

        selections = plan_selection(
                base_dir=DEFAULT_BASE,
                hours_threshold=HOURS_THRESHOLD,
                out_root=OUT_ROOT,
                copy_mode=COPY_MODE
                )

        if not selections:
                logging.warning("No selections found at or above threshold.")
        else:
                logging.info(f"Planned selections: {len(selections)}")
                execute_selection(selections, COPY_MODE, DRY_RUN)

        return 0


if __name__ == "__main__":
        main()