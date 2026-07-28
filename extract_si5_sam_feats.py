import os, sys
import numpy as np
import pandas as pd
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segment_anything import sam_model_registry, SamPredictor

SAM_CKP = "/mnt/hdd2/task2/sam/sam_vit_h_4b8939.pth"
FEAT_OUT = "/mnt/hdd2/task2/surgicalsam/dataset/feat_dir_Si5"
CSV_DIR = "/mnt/hdd2/task2/Si5/processed"

# Step 1: combine all 5 Si5 CSVs, get unique image paths
dfs = []
for f in range(1, 6):
    df = pd.read_csv(os.path.join(CSV_DIR, f"output_bbox_test_Si5_{f}.csv"))
    dfs.append(df)
combined = pd.concat(dfs, ignore_index=True)
unique_images = combined["png_filename"].unique()
print(f"Combined {len(combined)} rows, {len(unique_images)} unique images from 5 Si5 CSVs")

# Step 2: load SAM model
print("Loading SAM ViT-H...")
sam = sam_model_registry["vit_h"](checkpoint=SAM_CKP).cuda()
predictor = SamPredictor(sam)
print("SAM loaded.")

# Step 3: extract features for each unique image
for i, img_path in enumerate(unique_images):
    dir_part, fname = os.path.split(img_path)
    pid = os.path.basename(dir_part)
    out_dir = os.path.join(FEAT_OUT, pid)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fname.replace(".png", ".npy"))

    if os.path.exists(out_path):
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(unique_images)}] skip existing")
        continue

    image = cv2.imread(img_path)
    if image is None:
        print(f"  [{i+1}/{len(unique_images)}] WARN: cannot read {img_path}")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    feat = predictor.features.squeeze().permute(1, 2, 0).cpu().numpy()
    np.save(out_path, feat)

    if (i + 1) % 50 == 0:
        print(f"  [{i+1}/{len(unique_images)}] {os.path.basename(img_path)} feat={feat.shape}")

print(f"DONE. Extracted {len(unique_images)} features to {FEAT_OUT}")
