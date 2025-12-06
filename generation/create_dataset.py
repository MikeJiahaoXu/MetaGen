#!/usr/bin/env python3
"""
create_dataset.py

Builds a metasurface dataset (images + metadata) from one or more tab-separated
input files (defaults: gen_data0.txt and gen_data1.txt). Each 61-line block is
treated as one sample:
  * the shared h1_o/l1_o/l2_o/d1_o parameters drive a 98x240 binary mask
  * the 61 phase-shift (rad) entries become the label vector stored in "param"

Example:
    python create_dataset.py \
        --data-files ./gen_data0.txt ./gen_data1.txt \
        --output-dir ./processed_dataset \
        --image-dir images \
        --meta-path metadata.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np


def parse_numeric_line(line: str, line_no: int) -> Dict[str, float]:
    """Split a numeric data line into the columns we care about."""
    parts = line.split()
    if len(parts) < 13:
        raise ValueError(f"Line {line_no}: expected at least 13 columns, got {len(parts)} ({line})")

    return {
        "line_no": line_no,
        "h1_o": float(parts[0]),
        "l1_o": float(parts[1]),
        "l2_o": float(parts[2]),
        "d1_o": float(parts[3]),
        "freq": float(parts[8]),
        "phase": float(parts[9]),
    }


def iter_blocks(data_file: Path, block_size: int) -> Iterable[List[Dict[str, float]]]:
    """Yield consecutive blocks of `block_size` records from the data file."""
    block: List[Dict[str, float]] = []
    with data_file.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            block.append(parse_numeric_line(stripped, line_no))
            if len(block) == block_size:
                yield block
                block = []

    if block:
        raise ValueError(f"Incomplete block at EOF (expected {block_size} rows, got {len(block)}).")


def ensure_constant_params(block: List[Dict[str, float]]) -> Dict[str, float]:
    """Verify that h1_o/l1_o/l2_o/d1_o stay constant inside a block."""
    base = block[0]
    for key in ("h1_o", "l1_o", "l2_o", "d1_o"):
        for row in block[1:]:
            if not math.isclose(row[key], base[key], rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(
                    f"Parameter {key} changed within one block (lines "
                    f"{base['line_no']} vs {row['line_no']})."
                )
    return base


def clip_range(start: int, end: int, limit: int) -> tuple[int, int]:
    """Clamp pixel coordinates into [0, limit]."""
    start = max(0, min(limit, start))
    end = max(0, min(limit, end))
    return start, end

def is_reducible(arr: np.ndarray) -> bool:
    """Return True if every 2×2 block has identical values."""
    tl = arr[::2, ::2]
    tr = arr[::2, 1::2]
    bl = arr[1::2, ::2]
    br = arr[1::2, 1::2]
    return np.array_equal(tl, tr) and np.array_equal(tl, bl) and np.array_equal(tl, br)

def reduce_by_half(arr: np.ndarray) -> np.ndarray:
    return arr[::2, ::2]

def generate_structure_image(
    h1: float,
    l1: float,
    l2: float,
    d1: float,
    width: int,
    length: int,
    d2: float,
) -> np.ndarray:
    """Reproduce the provided 2-layer structure drawing logic."""
    img = np.zeros((length, width), dtype=np.uint8)

    # fixed side bars
    border_w = 6
    img[:, :border_w] = 255
    img[:, width - border_w :] = 255

    # convert all geometry to pixel counts
    mid1_w = int(round((width - 2 * d1 * 100 - 2 * d2 * 100 - h1 * 100) / 2))
    mid1_h = int(round((length - l1 * 100) / 2))
    mid2_w = int(round(d1 * 100))
    mid2_h = int(round((length - l2 * 100) / 2))

    mid1_w = max(mid1_w, 0)
    mid1_h = max(mid1_h, 0)
    mid2_w = max(mid2_w, 0)
    mid2_h = max(mid2_h, 0)

    left_mid1_x0, left_mid1_x1 = border_w, border_w + mid1_w
    left_mid2_x0, left_mid2_x1 = left_mid1_x1, left_mid1_x1 + mid2_w

    right_mid1_x1 = width - border_w
    right_mid1_x0 = right_mid1_x1 - mid1_w
    right_mid2_x1 = right_mid1_x0
    right_mid2_x0 = right_mid2_x1 - mid2_w

    top_mid1_y0, top_mid1_y1 = 0, mid1_h
    bot_mid1_y0, bot_mid1_y1 = length - mid1_h, length
    top_mid2_y0, top_mid2_y1 = 0, mid2_h
    bot_mid2_y0, bot_mid2_y1 = length - mid2_h, length

    left_mid1_x0, left_mid1_x1 = clip_range(left_mid1_x0, left_mid1_x1, width)
    left_mid2_x0, left_mid2_x1 = clip_range(left_mid2_x0, left_mid2_x1, width)
    right_mid1_x0, right_mid1_x1 = clip_range(right_mid1_x0, right_mid1_x1, width)
    right_mid2_x0, right_mid2_x1 = clip_range(right_mid2_x0, right_mid2_x1, width)
    top_mid1_y0, top_mid1_y1 = clip_range(top_mid1_y0, top_mid1_y1, length)
    bot_mid1_y0, bot_mid1_y1 = clip_range(bot_mid1_y0, bot_mid1_y1, length)
    top_mid2_y0, top_mid2_y1 = clip_range(top_mid2_y0, top_mid2_y1, length)
    bot_mid2_y0, bot_mid2_y1 = clip_range(bot_mid2_y0, bot_mid2_y1, length)

    # fill rectangles (four per layer)
    if left_mid1_x1 > left_mid1_x0 and top_mid1_y1 > top_mid1_y0:
        img[top_mid1_y0:top_mid1_y1, left_mid1_x0:left_mid1_x1] = 255
    if left_mid1_x1 > left_mid1_x0 and bot_mid1_y1 > bot_mid1_y0:
        img[bot_mid1_y0:bot_mid1_y1, left_mid1_x0:left_mid1_x1] = 255
    if right_mid1_x1 > right_mid1_x0 and top_mid1_y1 > top_mid1_y0:
        img[top_mid1_y0:top_mid1_y1, right_mid1_x0:right_mid1_x1] = 255
    if right_mid1_x1 > right_mid1_x0 and bot_mid1_y1 > bot_mid1_y0:
        img[bot_mid1_y0:bot_mid1_y1, right_mid1_x0:right_mid1_x1] = 255

    if left_mid2_x1 > left_mid2_x0 and top_mid2_y1 > top_mid2_y0:
        img[top_mid2_y0:top_mid2_y1, left_mid2_x0:left_mid2_x1] = 255
    if left_mid2_x1 > left_mid2_x0 and bot_mid2_y1 > bot_mid2_y0:
        img[bot_mid2_y0:bot_mid2_y1, left_mid2_x0:left_mid2_x1] = 255
    if right_mid2_x1 > right_mid2_x0 and top_mid2_y1 > top_mid2_y0:
        img[top_mid2_y0:top_mid2_y1, right_mid2_x0:right_mid2_x1] = 255
    if right_mid2_x1 > right_mid2_x0 and bot_mid2_y1 > bot_mid2_y0:
        img[bot_mid2_y0:bot_mid2_y1, right_mid2_x0:right_mid2_x1] = 255

    if is_reducible(img):
        return reduce_by_half(img)
    else:
        print("Warning: image is not reducible; returning original.")
        return img


def save_image(img: np.ndarray, path: Path) -> None:
    """Persist a grayscale PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, img, cmap="gray", vmin=0, vmax=255)


def build_dataset(args: argparse.Namespace) -> None:
    """Main orchestration: parse file(s), create images, and write metadata."""
    data_files = [Path(p).resolve() for p in args.data_files]
    output_dir = Path(args.output_dir).resolve()
    image_dir = (output_dir / args.image_dir).resolve()
    meta_path = Path(args.meta_path)
    if not meta_path.is_absolute():
        meta_path = output_dir / meta_path
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    with meta_path.open("w", encoding="utf-8") as meta_fp:
        for data_file in data_files:
            for block in iter_blocks(data_file, args.block_size):
                base = ensure_constant_params(block)
                img = generate_structure_image(
                    h1=base["h1_o"],
                    l1=base["l1_o"],
                    l2=base["l2_o"],
                    d1=base["d1_o"],
                    width=args.width,
                    length=args.length,
                    d2=args.d2,
                )
                file_name = f"{args.prefix}_{total_samples:06d}.png"
                save_image(img, image_dir / file_name)

                phases = [row["phase"] for row in block]
                freqs = [row["freq"] for row in block]

                meta_record = {
                    "file_name": file_name,
                    "param": phases,
                    "condition": {
                        "h1_o": base["h1_o"],
                        "l1_o": base["l1_o"],
                        "l2_o": base["l2_o"],
                        "d1_o": base["d1_o"],
                        "frequencies_hz": freqs,
                    },
                }
                meta_fp.write(json.dumps(meta_record) + "\n")
                total_samples += 1

    print(f"Wrote {total_samples} samples.")
    print(f"Images: {image_dir}")
    print(f"Metadata: {meta_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert tab-separated data files into images + JSONL metadata."
    )
    parser.add_argument(
        "--data-files",
        nargs="+",
        default=["gen_data0.txt", "gen_data1.txt"],
        help="Path(s) to tab-separated data source files; all will be combined.",
    )
    parser.add_argument(
        "--output-dir",
        default="test_dataset",
        help="Root directory for generated artifacts (images + metadata).",
    )
    parser.add_argument(
        "--image-dir",
        default="images",
        help="Subdirectory (relative to --output-dir) where PNGs will be stored.",
    )
    parser.add_argument(
        "--meta-path",
        default="metadata.jsonl",
        help="Output metadata file. Relative paths are resolved inside --output-dir.",
    )
    parser.add_argument("--block-size", type=int, default=61, help="Rows per sample.")
    parser.add_argument("--width", type=int, default=96, help="Image width in pixels.")
    parser.add_argument("--length", type=int, default=240, help="Image height in pixels.")
    parser.add_argument("--d2", type=float, default=0.05, help="Gap parameter used in the template.")
    parser.add_argument(
        "--prefix",
        default="sample",
        help="Filename prefix for generated PNGs (suffix is zero-padded index).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
