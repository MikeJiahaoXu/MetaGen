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

    # 左右各一列白色（255）
    left_col  = np.full((h, 3), 255, dtype=np.uint8)
    right_col = np.full((h, 3), 255, dtype=np.uint8)

    # 拼接：左白 + 原图 + 右白
    new_arr = np.concatenate([left_col, arr, right_col], axis=1)

    new_img = Image.fromarray(new_arr)
    new_img.save(path)   # 覆盖保存

print("Done.")
