import os
from PIL import Image
import numpy as np

import sys

part = int(sys.argv[1])   # 0 or 1

folder = "test_dataset/images"

files = sorted(os.listdir(folder))
start = part * 5000
end = start + 5000
files = files[start:end]

for fn in files:

    if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    path = os.path.join(folder, fn)
    img = Image.open(path).convert("L")   # 保持单通道/灰度（如果是RGB改成 "RGB"）
    arr = np.array(img)

    h, w = arr.shape  # should be 240 x 98

    # 左右各裁掉 4 列，宽度减少 8 像素
    new_arr = arr[:, 1:-1]

    new_img = Image.fromarray(new_arr)
    new_img.save(path)   # 覆盖保存

print("Done.")
