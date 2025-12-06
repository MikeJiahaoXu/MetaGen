import os
from typing import List, Tuple

import numpy as np
from PIL import Image


def is_reducible(arr: np.ndarray) -> bool:
    """Return True if every 2x2 block is the same value."""
    if arr.ndim == 2:
        tl = arr[::2, ::2]
        tr = arr[::2, 1::2]
        bl = arr[1::2, ::2]
        br = arr[1::2, 1::2]
    elif arr.ndim == 3:
        tl = arr[::2, ::2, :]
        tr = arr[::2, 1::2, :]
        bl = arr[1::2, ::2, :]
        br = arr[1::2, 1::2, :]
    else:
        return False

    return np.array_equal(tl, tr) and np.array_equal(tl, bl) and np.array_equal(tl, br)


def main() -> None:
    folder = "test_dataset/images"
    bad_files: List[str] = []
    images: List[Tuple[str, np.ndarray, str, str]] = []  # path, array, mode, format

    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        if fn.lower().endswith("sample_009912.png"):
            continue

        path = os.path.join(folder, fn)
        with Image.open(path) as img:
            arr = np.array(img)
            h, w = arr.shape[:2]

            # Require even dimensions for 2x2 reduction.
            if h % 2 or w % 2:
                bad_files.append(fn)
                continue

            if is_reducible(arr):
                images.append((path, arr, img.mode, img.format))
            else:
                bad_files.append(fn)

    if bad_files:
        report_path = os.path.join(folder, "non_reducible.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(bad_files))
        print(f"Found {len(bad_files)} non-reducible images. Listed in {report_path}.")
        return

    for path, arr, mode, fmt in images:
        reduced = arr[::2, ::2] if arr.ndim == 2 else arr[::2, ::2, :]
        out = Image.fromarray(reduced, mode=mode)
        out.save(path, format=fmt)

    print(f"Reduced {len(images)} images to 120x48.")


if __name__ == "__main__":
    main()
