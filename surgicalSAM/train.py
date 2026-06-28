"""
data: /mnt/hdd2/task2/surgicalsam/
    dataset/folds/{train,val}_fold{0..4}.csv, dataset/test.csv
    dataset/{feat_dir,binary_annotations,class_embeddings}
    work_dirs/fold_X/
    result/predict_masks/fold_X/

SAM ckp: ../ckp/sam/sam_vit_h_4b8939.pth  (

训练命令 (以fold0为例):
source /home/lq/Projects_qin/run_SSAM/bin/activate
cd /home/lq/Projects_qin/surgical_semantic_seg/benmarking_algorithms/SurgicalSAM/surgicalSAM
mkdir -p /mnt/hdd2/task2/surgicalsam/work_dirs/fold_0
CUDA_VISIBLE_DEVICES=1 nohup /home/lq/Projects_qin/run_SSAM/bin/python train.py \
      --fold 0 \
      > /mnt/hdd2/task2/surgicalsam/work_dirs/fold_0/train.out 2>&1 &

参数是: --epochs 100 --patience 5 --batch_size 2 --lr 0.001 --num_tokens 2

推理命令 (以fold0为例):
CUDA_VISIBLE_DEVICES=1 /home/lq/Projects_qin/run_SSAM/bin/python inference.py --fold 0
# -> result/predict_masks/fold_0/{pid}_{frame}.png  +  work_dirs/fold_0/SSAM_test_results.csv

输出结构:
/mnt/hdd2/task2/surgicalsam/work_dirs/fold_X/
├── log.txt
├── model_ckp.pth
├── training_metrics.png
└── SSAM_test_results.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os.path as osp
import argparse
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from segment_anything import sam_model_registry
from dataset import SurgicalInternalDataset, read_gt_masks_internal
from model import Learnable_Prototypes, Prototype_Prompt_Encoder
from model_forward import model_forward_function
from utils import print_log, EndovisMaskAccumulator, eval_endovis
from loss import DiceLoss
from pytorch_metric_learning import losses
from medpy.metric.binary import hd95
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAM_CKP = "../ckp/sam/sam_vit_h_4b8939.pth"   # symlink -> /mnt/hdd2/task2/sam/...
SURG = "/mnt/hdd2/task2/surgicalsam"           # canonical home: dataset/ + work_dirs/
FOLDS_DIR = f"{SURG}/dataset/folds"
NUM_CLASSES = 28
FRAME_H, FRAME_W = 1004, 1920


def load_fold_df(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["label_id"] <= NUM_CLASSES].reset_index(drop=True)  # drop Background (29)
    return df


def load_models(num_tokens):
    # no_image_encoder variant: only prompt encoder + mask decoder live on GPU
    sam_prompt_encoder, sam_decoder = sam_model_registry["vit_h_no_image_encoder"](checkpoint=SAM_CKP)
    sam_prompt_encoder.cuda()
    sam_decoder.cuda()
    for p in sam_prompt_encoder.parameters():
        p.requires_grad = False
    for p in sam_decoder.parameters():
        p.requires_grad = True

    prototypes_model = Learnable_Prototypes(num_classes=NUM_CLASSES, feat_dim=256).cuda()
    encoder = Prototype_Prompt_Encoder(feat_dim=256, hidden_dim_dense=128,
                                       hidden_dim_sparse=128, size=64,
                                       num_tokens=num_tokens).cuda()

    # seed the +/- class embeddings from SAM's point embeddings (original recipe)
    state_dict = torch.load(SAM_CKP)
    pn = {k.split("prompt_encoder.point_embeddings.")[-1]: v
          for k, v in state_dict.items()
          if k.startswith("prompt_encoder.point_embeddings") and ("0" in k or "1" in k)}
    pn_ckp = {"0.weight": torch.concat([pn["0.weight"]] * num_tokens, dim=0),
              "1.weight": torch.concat([pn["1.weight"]] * num_tokens, dim=0)}
    encoder.pn_cls_embeddings.load_state_dict(pn_ckp)

    for p in prototypes_model.parameters():
        p.requires_grad = True
    for name, p in encoder.named_parameters():
        p.requires_grad = "pn_cls_embeddings" not in name

    return sam_prompt_encoder, sam_decoder, prototypes_model, encoder


def val_overlap_metrics(endo, gt):
    """Mean IoU / Dice / HD95 over the validation set, averaged across every
    (frame, gt-class) pair (background and class 29 excluded)."""
    ious, dices, hds = [], [], []
    for fname, pred in endo.items():
        target = gt[fname].numpy()
        classes = np.unique(target)
        classes = classes[(classes > 0) & (classes != 29)]
        for c in classes:
            c = int(c)
            p = (pred == c); t = (target == c)
            inter = float(np.logical_and(p, t).sum())
            psum = float(p.sum()); tsum = float(t.sum()); uni = psum + tsum - inter
            ious.append(inter / uni if uni else 0.0)
            dices.append(2 * inter / (psum + tsum) if (psum + tsum) else 0.0)
            hds.append(0.0 if (psum == 0 or tsum == 0)
                       else hd95(p.astype(np.float32), t.astype(np.float32)))
    return float(np.mean(ious)), float(np.mean(dices)), float(np.mean(hds))


def plot_training_metrics(history, out_png, fold):
    """2x2 training-curve figure: Loss / IoU / Dice+IoU / HD95 vs epoch."""
    ep = [h["epoch"] for h in history]
    g = lambda k: [h[k] for h in history]
    fig, ax = plt.subplots(2, 2, figsize=(16, 12))

    ax[0, 0].plot(ep, g("train_loss"), "-o", color="#1f77b4", label="Training Mean Loss")
    ax[0, 0].plot(ep, g("val_loss"), "-s", color="#ff7f0e", label="Validation Mean Loss")
    ax[0, 0].set_title("Training and Validation Mean Loss"); ax[0, 0].set_ylabel("Loss")

    ax[0, 1].plot(ep, g("train_iou"), "-o", color="#ffa500", label="Training Mean IoU")
    ax[0, 1].plot(ep, g("val_iou"), "-s", color="#d62728", label="Validation Mean IoU")
    ax[0, 1].set_title("Training and Validation Mean IoU"); ax[0, 1].set_ylabel("IoU")

    ax[1, 0].plot(ep, g("val_dice"), "-s", color="#1f77b4", label="Validation Mean Dice")
    ax[1, 0].plot(ep, g("val_iou"), "-s", color="#ff7f0e", label="Validation Mean IoU")
    ax[1, 0].set_title("Validation Mean Dice and Mean IoU"); ax[1, 0].set_ylabel("Score")

    ax[1, 1].plot(ep, g("val_hd95"), "-s", color="#d62728", label="Validation Mean HD95")
    ax[1, 1].set_title("Validation Mean HD95"); ax[1, 1].set_ylabel("HD95")

    for a in ax.ravel():
        a.set_xlabel("Epoch"); a.grid(True, alpha=0.4); a.legend()
    fig.suptitle(f"SurgicalSAM fold {fold} training curves", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--num_tokens", type=int, default=2)
    ap.add_argument("--val_every", type=int, default=1)
    ap.add_argument("--max_iters", type=int, default=0)
    ap.add_argument("--save_root", default=f"{SURG}/work_dirs")
    args = ap.parse_args()

    seed = 301
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)

    fold = args.fold
    save_dir = osp.join(args.save_root, f"fold_{fold}")
    os.makedirs(save_dir, exist_ok=True)
    log_file = osp.join(save_dir, "log.txt")
    print_log(f"==== SurgicalSAM internal fold{fold} ==== {vars(args)}", log_file)

    train_df = load_fold_df(f"{FOLDS_DIR}/train_fold{fold}.csv")
    val_df = load_fold_df(f"{FOLDS_DIR}/val_fold{fold}.csv")
    print_log(f"train rows {len(train_df)} | val rows {len(val_df)}", log_file)

    train_loader = DataLoader(SurgicalInternalDataset(train_df), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(SurgicalInternalDataset(val_df), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print_log("caching val GT ...", log_file)
    gt_val = read_gt_masks_internal(val_df["file_name"].tolist(), H=FRAME_H)
    print_log(f"cached {len(gt_val)} val GT frames", log_file)

    sam_pe, sam_dec, proto_model, encoder = load_models(args.num_tokens)
    seg_loss_fn = DiceLoss().cuda()
    contr_loss_fn = losses.NTXentLoss(temperature=0.07).cuda()
    optim = torch.optim.Adam([
        {"params": proto_model.parameters()},
        {"params": encoder.parameters()},
        {"params": sam_dec.parameters()},
    ], lr=args.lr, weight_decay=1e-4)

    best = -1.0
    no_improve = 0                     # consecutive validations without improvement
    history = []                       # per-epoch metrics for training_metrics.png
    for epoch in range(args.epochs):
        encoder.train(); sam_dec.train(); proto_model.train()
        t0 = time.time(); running = 0.0; nb = 0
        tr_inter = torch.zeros((), device="cuda"); tr_uni = torch.zeros((), device="cuda")
        for it, (feats, _, cls_ids, masks, cembs) in enumerate(train_loader):
            if args.max_iters and it >= args.max_iters:
                break
            feats = feats.cuda(non_blocking=True)
            cls_ids = cls_ids.cuda(non_blocking=True)
            masks = masks.cuda(non_blocking=True)
            cembs = cembs.cuda(non_blocking=True)

            prototypes = proto_model()
            preds, _ = model_forward_function(encoder, sam_pe, sam_dec, feats, prototypes, cls_ids)

            contr = contr_loss_fn(prototypes,
                                  torch.arange(1, prototypes.size(0) + 1).cuda(),
                                  ref_emb=cembs, ref_labels=cls_ids)
            seg = seg_loss_fn(preds, masks / 255.0)
            loss = seg + contr

            optim.zero_grad(); loss.backward(); optim.step()
            running += loss.item(); nb += 1
            with torch.no_grad():           # running training IoU (foreground pred>0 vs mask>0)
                pb = preds > 0; mb = masks > 0
                tr_inter += (pb & mb).sum(); tr_uni += (pb | mb).sum()
            if it % 200 == 0:
                print_log(f"[f{fold}] ep{epoch} it{it}/{len(train_loader)} loss={loss.item():.4f} "
                          f"(seg={seg.item():.4f} contr={contr.item():.4f}) "
                          f"{(it + 1) / (time.time() - t0):.1f}it/s", log_file)

        train_loss = running / max(nb, 1)
        train_iou = (tr_inter / tr_uni.clamp(min=1)).item()
        print_log(f"[f{fold}] epoch {epoch} mean_loss={train_loss:.4f} train_iou={train_iou:.4f} "
                  f"time={(time.time() - t0) / 60:.1f}min", log_file)

        # ---- validation ----
        if ((epoch + 1) % args.val_every) and epoch != args.epochs - 1:
            continue
        encoder.eval(); sam_dec.eval(); proto_model.eval()
        binary_acc = EndovisMaskAccumulator(FRAME_H, FRAME_W); val_running = 0.0; vnb = 0
        with torch.no_grad():
            prototypes = proto_model()
            for vi, (feats, mask_names, cls_ids, masks, cembs) in enumerate(val_loader):
                if args.max_iters and vi >= args.max_iters:
                    break
                feats = feats.cuda(non_blocking=True); cls_ids = cls_ids.cuda(non_blocking=True)
                masks = masks.cuda(non_blocking=True); cembs = cembs.cuda(non_blocking=True)
                preds, q = model_forward_function(encoder, sam_pe, sam_dec, feats, prototypes, cls_ids)
                contr = contr_loss_fn(prototypes, torch.arange(1, prototypes.size(0) + 1).cuda(),
                                      ref_emb=cembs, ref_labels=cls_ids)
                val_running += (seg_loss_fn(preds, masks / 255.0) + contr).item(); vnb += 1
                binary_acc.add(preds, q, mask_names, thr=0)
        endo = binary_acc.result()
        res = eval_endovis(endo, gt_val)
        val_loss = val_running / max(vnb, 1)
        val_iou, val_dice, val_hd95 = val_overlap_metrics(endo, gt_val)
        print_log(f"[f{fold}] VAL epoch {epoch}: {res}", log_file)
        print_log(f"[f{fold}] VAL epoch {epoch}: val_loss={val_loss:.4f} IoU={val_iou:.4f} "
                  f"Dice={val_dice:.4f} HD95={val_hd95:.2f}", log_file)

        history.append({"epoch": epoch, "train_loss": train_loss, "train_iou": train_iou,
                        "val_loss": val_loss, "val_iou": val_iou, "val_dice": val_dice,
                        "val_hd95": val_hd95, "challengIoU": res["challengIoU"]})

        if res["challengIoU"] > best:
            best = res["challengIoU"]
            no_improve = 0
            torch.save({
                "prototype_prompt_encoder_state_dict": encoder.state_dict(),
                "sam_decoder_state_dict": sam_dec.state_dict(),
                "prototypes_state_dict": proto_model.state_dict(),
                "epoch": int(epoch), "challengIoU": float(best),
            }, osp.join(save_dir, "model_ckp.pth"))
            print_log(f"[f{fold}] *** best challengeIoU {best:.4f} @ epoch {epoch} -> saved", log_file)
        else:
            no_improve += 1
            print_log(f"[f{fold}] no improvement {no_improve}/{args.patience} (best={best:.4f})", log_file)

        # ---- early stopping ----
        if args.patience and no_improve >= args.patience:
            print_log(f"[f{fold}] EARLY STOP at epoch {epoch}: "
                      f"challengeIoU not improved for {no_improve} validations (best={best:.4f})", log_file)
            break

    # draw the training-curve figure ONCE, only after training fully finishes
    # (early stop triggered or max epochs reached)
    if history:
        plot_training_metrics(history, osp.join(save_dir, "training_metrics.png"), fold)
    print_log(f"[f{fold}] DONE. best challengeIoU={best:.4f} "
              f"-> {osp.join(save_dir, 'training_metrics.png')}", log_file)


if __name__ == "__main__":
    main()
