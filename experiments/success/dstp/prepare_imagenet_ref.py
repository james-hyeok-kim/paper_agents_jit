"""
ImageNet val에서 2000장 균등 샘플링하여 FID reference set 구성.
PixelDiT-XL이 256x256 PNG로 생성하므로 ref도 동일 포맷으로 정렬.
"""
import os
import random
from PIL import Image
import cv2
import numpy as np

VAL_DIR = "/data/imagenet/val"
OUT_DIR = "/data/jameskimh/imagenet_val_ref_2k"
N_PER_CLASS = 2  # 2 per class * 1000 classes = 2000
SIZE = 256

os.makedirs(OUT_DIR, exist_ok=True)

classes = sorted(os.listdir(VAL_DIR))
print(f"클래스 수: {len(classes)}")

random.seed(0)
count = 0
for cls in classes:
    cls_dir = os.path.join(VAL_DIR, cls)
    files = sorted(os.listdir(cls_dir))
    picked = random.sample(files, min(N_PER_CLASS, len(files)))

    for fname in picked:
        src = os.path.join(cls_dir, fname)
        try:
            img = Image.open(src).convert("RGB")
        except Exception as e:
            print(f"  skip {fname}: {e}")
            continue

        # center crop + resize to 256
        W, H = img.size
        s = min(W, H)
        left = (W - s) // 2
        top = (H - s) // 2
        img = img.crop((left, top, left + s, top + s))
        img = img.resize((SIZE, SIZE), Image.BICUBIC)

        out_path = os.path.join(OUT_DIR, f"{count:05d}.png")
        # PixelDiT 샘플과 같은 형식: cv2 imwrite로 PNG 저장 (BGR)
        arr = np.array(img)[:, :, ::-1]  # RGB → BGR
        cv2.imwrite(out_path, arr)
        count += 1

        if count % 200 == 0:
            print(f"  {count}/{N_PER_CLASS * len(classes)}")

print(f"\n완료: {count}장 -> {OUT_DIR}")
