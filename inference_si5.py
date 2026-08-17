"""
SurgicalSAM Si5 inference — produces native (1004,1920) integer semantic PNGs per fold.
Usage: python inference_si5.py --fold 0 [--num_tokens 2] [--num_workers 4]
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "surgicalSAM"))
import os.path as osp
import re
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import cv2

from segment_anything import sam_model_registry
from model import Learnable_Prototypes, Prototype_Prompt_Encoder
from model_forward import model_forward_function
from utils import EndovisMaskAccumulator, eval_endovis

SAM_CKP = "/mnt/hdd2/task2/sam/sam_vit_h_4b8939.pth"
SURG = "/mnt/hdd2/task2/surgicalsam"
DATA_ROOT = f"{SURG}/dataset"
FEAT_DIR_SI5 = f"{DATA_ROOT}/feat_dir_Si5"         # pre-extracted SAM features for Si5
NUM_CLASSES = 28
FRAME_H, FRAME_W = 1004, 1920

# ---- Si5-specific dataset ---------------------------------------------------
class SurgicalSi5Dataset(Dataset):
    """Like SurgicalInternalDataset but reads features from FEAT_DIR_SI5."""
    def __init__(self, df):
        self.png = df["png_filename"].tolist()
        self.cls = df["label_id"].tolist()
        self.bm  = df["binary_annotations"].tolist()

    def __len__(self):
        return len(self.png)

    def __getitem__(self, idx):
        cls_id  = int(self.cls[idx])
        bm_path = self.bm[idx]

        # ── SAM feature: FEAT_DIR_SI5 / {pid} / {frame}.npy ──
        img_path = self.png[idx]
        dir_part, fname = osp.split(img_path)
        pid = osp.basename(dir_part)
        feat_path = osp.join(FEAT_DIR_SI5, pid, fname.replace(".png", ".npy"))
        sam_feat = np.load(feat_path)

        # ── binary mask (read but discarded during inference; needed for Dataset contract) ──
        mask = cv2.imread(bm_path, cv2.IMREAD_GRAYSCALE)

        # mask name as "pid/C1-XXXXX_classN.png" → endo_key_to_nnkey → nnUNet key
        # (strip patient prefix from Si5 binary annotation basename: "1_C1-00000_class15.png" -> "C1-00000_class15.png")
        bm_basename = osp.basename(bm_path)
        if "_" in bm_basename:
            bm_basename = bm_basename.split("_", 1)[1]  # remove "pid_" prefix
        pid_mask = f"{pid}/{bm_basename}"

        # class embedding (zeros fallback)
        ce_path = osp.join(DATA_ROOT, "class_embeddings",
                           pid_mask.replace(".png", ".npy"))
        try:
            class_embedding = np.load(ce_path)
        except Exception:
            class_embedding = np.zeros((256,), dtype=np.float32)

        return sam_feat, pid_mask, cls_id, mask, class_embedding


# ---- helpers ----------------------------------------------------------------
def load_models(ckp_path, num_tokens):
    sam_prompt_encoder, sam_decoder = sam_model_registry["vit_h_no_image_encoder"](checkpoint=SAM_CKP)
    sam_prompt_encoder.cuda()
    sam_decoder.cuda()
    proto = Learnable_Prototypes(num_classes=NUM_CLASSES, feat_dim=256).cuda()
    enc = Prototype_Prompt_Encoder(feat_dim=256, hidden_dim_dense=128,
                                   hidden_dim_sparse=128, size=64,
                                   num_tokens=num_tokens).cuda()
    ckp = torch.load(ckp_path, weights_only=False)
    enc.load_state_dict(ckp["prototype_prompt_encoder_state_dict"])
    sam_decoder.load_state_dict(ckp["sam_decoder_state_dict"])
    proto.load_state_dict(ckp["prototypes_state_dict"])
    print(f"loaded {ckp_path} (epoch={ckp.get('epoch')}, val challengeIoU={ckp.get('challengIoU')})")
    for m in (sam_prompt_encoder, sam_decoder, proto, enc):
        for p in m.parameters():
            p.requires_grad = False
    enc.eval(); sam_decoder.eval(); proto.eval()
    return sam_prompt_encoder, sam_decoder, proto, enc


def endo_key_to_nnkey(k):
    pid  = k.split("/")[0]
    base = k.split("/")[-1].replace(".png", "")
    m = re.match(r"C(\d+)-(\d+)", base)
    return f"{pid}_{m.group(2)}" if m else f"{pid}_{base}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4])
    ap.add_argument("--num_tokens", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--save_root", default=f"{SURG}/work_dirs_bs32")
    ap.add_argument("--predict_dir", default=None,
                    help=f"default: {SURG}/result/predict_masks_bs32_Si5/fold_<fold>")
    args = ap.parse_args()

    ckp = osp.join(args.save_root, f"fold_{args.fold}", "model_ckp.pth")
    predict_dir = args.predict_dir or osp.join(SURG, "result", "predict_masks_bs32_Si5", f"fold_{args.fold}")
    os.makedirs(predict_dir, exist_ok=True)

    # ── load combined Si5 CSV ──
    csv_dir = "/mnt/hdd2/task2/Si5/processed"
    dfs = []
    for f in range(1, 6):
        dfs.append(pd.read_csv(osp.join(csv_dir, f"output_bbox_test_Si5_{f}.csv")))
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["label_id"] <= NUM_CLASSES].reset_index(drop=True)
    print(f"[fold{args.fold}] Si5 CSV: {len(df)} rows")

    sam_pe, sam_dec, proto, enc = load_models(ckp, args.num_tokens)
    loader = DataLoader(SurgicalSi5Dataset(df), batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    binary_acc = EndovisMaskAccumulator(FRAME_H, FRAME_W)
    with torch.no_grad():
        prototypes = proto()
        for feats, mask_names, cls_ids, _, _ in loader:
            feats = feats.cuda(non_blocking=True)
            cls_ids = cls_ids.cuda(non_blocking=True)
            preds, q = model_forward_function(enc, sam_pe, sam_dec, feats, prototypes, cls_ids)
            binary_acc.add(preds, q, mask_names, thr=0)

    endo = binary_acc.result()

    # save native (1004,1920) integer semantic PNGs
    for fname, pred in endo.items():
        nnkey = endo_key_to_nnkey(fname)
        Image.fromarray(pred.astype(np.uint8)).save(osp.join(predict_dir, f"{nnkey}.png"))
    print(f"[fold{args.fold}] saved {len(endo)} native semantic PNGs -> {predict_dir}")

    print(f"[fold{args.fold}] done. pred frames={len(endo)}")


if __name__ == "__main__":
    main()
