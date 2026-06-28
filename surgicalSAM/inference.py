import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os.path as osp
import re
import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from medpy.metric.binary import hd95

from segment_anything import sam_model_registry
from dataset import SurgicalInternalDataset, read_gt_masks_internal
from model import Learnable_Prototypes, Prototype_Prompt_Encoder
from model_forward import model_forward_function
from utils import EndovisMaskAccumulator, eval_endovis

SAM_CKP = "../ckp/sam/sam_vit_h_4b8939.pth"
SURG = "/mnt/hdd2/task2/surgicalsam"
TEST_CSV = f"{SURG}/dataset/test.csv"
NUM_CLASSES = 28
FRAME_H, FRAME_W = 1004, 1920


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


def compute_iou_dice_hd(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    psum = pred.sum(); tsum = gt.sum(); uni = psum + tsum - inter
    iou = inter / uni if uni else 0.0
    dice = 2 * inter / (psum + tsum) if (psum + tsum) else 0.0
    # empty pred or empty gt -> HD95 undefined -> NaN (unified convention, item 4)
    hd = np.nan if (psum == 0 or tsum == 0) else round(float(hd95(pred.astype(np.float32), gt.astype(np.float32))), 4)
    return round(float(iou), 4), round(float(dice), 4), hd


def endo_key_to_nnkey(k):
    """endo/GT key '19/C1-00000.png'  ->  nnUNet key '19_00000' (drops clip id)."""
    pid = k.split("/")[0]
    base = k.split("/")[-1].replace(".png", "")
    m = re.match(r"C(\d+)-(\d+)", base)
    return f"{pid}_{m.group(2)}" if m else f"{pid}_{base}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4])
    ap.add_argument("--num_tokens", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--ckp", default=None, help="override checkpoint path")
    ap.add_argument("--out", default=None, help="override output csv path")
    ap.add_argument("--save_root", default=f"{SURG}/work_dirs")
    ap.add_argument("--predict_dir", default=None,
                    help="dir to save native (1004x1920) integer semantic PNGs for the unified eval "
                         f"(default: {SURG}/result/predict_masks/fold_<fold>)")
    ap.add_argument("--limit", type=int, default=0, help="debug: only run first N test rows (0=all)")
    args = ap.parse_args()

    ckp = args.ckp or osp.join(args.save_root, f"fold_{args.fold}", "model_ckp.pth")
    out = args.out or osp.join(args.save_root, f"fold_{args.fold}", "SSAM_test_results.csv")
    os.makedirs(osp.dirname(out), exist_ok=True)

    df = pd.read_csv(TEST_CSV)
    df = df[df["label_id"] <= NUM_CLASSES].reset_index(drop=True)
    if args.limit:
        df = df.iloc[:args.limit].reset_index(drop=True)

    sam_pe, sam_dec, proto, enc = load_models(ckp, args.num_tokens)
    loader = DataLoader(SurgicalInternalDataset(df), batch_size=1, shuffle=False,
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
    gt = read_gt_masks_internal(df["file_name"].tolist(), H=FRAME_H)

    # save native (1004,1920) integer semantic predictions named by nnUNet key
    # ({pid}_{frame}.png) so the unified pipeline can score SurgicalSAM exactly
    # like the other native_map baselines (nnunet / longiseg). (item 5)
    predict_dir = args.predict_dir or osp.join(SURG, "result", "predict_masks", f"fold_{args.fold}")
    os.makedirs(predict_dir, exist_ok=True)
    for fname, pred in endo.items():
        nnkey = endo_key_to_nnkey(fname)
        Image.fromarray(pred.astype(np.uint8)).save(osp.join(predict_dir, f"{nnkey}.png"))
    print(f"[fold{args.fold}] saved {len(endo)} native semantic PNGs -> {predict_dir}")

    res = eval_endovis(endo, gt)
    print(f"[fold{args.fold}] Overall test: {res}")
    pd.DataFrame([res]).to_csv(out.replace(".csv", "_overall.csv"), index=False)

    rows = []
    for fname, pred in endo.items():
        target = gt[fname].numpy()
        classes = np.unique(target)
        classes = classes[(classes > 0) & (classes != 29)]
        for c in classes:
            c = int(c)
            iou, dice, hd = compute_iou_dice_hd((pred == c).astype(np.float64),
                                                (target == c).astype(np.float64))
            rows.append({"png_filename": fname, "label_id": c,
                         "iou_value": iou, "dice_value": dice, "hd95_value": hd})
    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(out, index=False)
    print(f"[fold{args.fold}] saved per-class results -> {out} ({len(rows)} rows)")
    print(f"[fold{args.fold}] MEAN  IoU={df_rows.iou_value.mean():.4f}  "
          f"Dice={df_rows.dice_value.mean():.4f}  HD95={df_rows.hd95_value.mean():.4f}")


if __name__ == "__main__":
    main()
