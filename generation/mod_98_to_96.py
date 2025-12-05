import os
from PIL import Image
import numpy as np

folder = "test_dataset/images"

for fn in os.listdir(folder):
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
