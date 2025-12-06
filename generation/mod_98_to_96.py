import os
from PIL import Image
import numpy as np

folder = "test_dataset/images"

for fn in os.listdir(folder):
    if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    path = os.path.join(folder, fn)
    img = Image.open(path).convert("L")
    arr = np.array(img)

    h, w = arr.shape  # should be 240×98 or 240×96

    if (h, w) == (240, 96):
        # already correct
        continue

    elif (h, w) == (240, 98):
        new_arr = arr[:, 1:-1]
        try:
            Image.fromarray(new_arr).save(path)
        except Exception as e:
            print(f"Failed to save {path}: {e}")


    else:
        print(f"Warning: {fn} has unexpected size {h}x{w}")

print("Done.")
