from torch.utils.data import Dataset
import os
import os.path as osp
import re
import numpy as np
import cv2
import torch

class Endovis18Dataset(Dataset):
    def __init__(self, data_root_dir = "../data/endovis_2018",
                 mode = "val",
                 vit_mode = "h",
                 version = 0):

        """Define the Endovis18 dataset

        Args:
            data_root_dir (str, optional): root dir containing all data for Endovis18. Defaults to "../data/endovis_2018".
            mode (str, optional): either in "train" or "val" mode. Defaults to "val".
            vit_mode (str, optional): "h", "l", "b" for huge, large, and base versions of SAM. Defaults to "h".
            version (int, optional): augmentation version to use. Defaults to 0.
        """

        self.vit_mode = vit_mode

        # directory containing all binary annotations
        if mode == "train":
            self.mask_dir = osp.join(data_root_dir, mode, str(version), "binary_annotations")
        elif mode == "val":
            self.mask_dir = osp.join(data_root_dir, mode, "binary_annotations")

        # put all binary masks into a list
        self.mask_list = []
        for subdir, _, files in os.walk(self.mask_dir):
            if len(files) == 0:
                continue
            self.mask_list += [osp.join(osp.basename(subdir),i) for i in files]

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, index):
        mask_name = self.mask_list[index]

        # get class id from mask_name
        cls_id = int(re.search(r"class(\d+)", mask_name).group(1))

        # get pre-computed sam feature
        feat_dir = osp.join(self.mask_dir.replace("binary_annotations", f"sam_features_{self.vit_mode}"), mask_name.split("_")[0] + ".npy")
        sam_feat = np.load(feat_dir)

        # get ground-truth mask
        mask_path = osp.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # get class embedding
        class_embedding_path = osp.join(self.mask_dir.replace("binary_annotations", f"class_embeddings_{self.vit_mode}"), mask_name.replace("png","npy"))
        class_embedding = np.load(class_embedding_path)

        return sam_feat, mask_name, cls_id, mask, class_embedding


class Endovis17Dataset(Dataset):
    def __init__(self, data_root_dir = "../data/endovis_2017",
                 mode = "val",
                 fold = 0,
                 vit_mode = "h",
                 version = 0):

        self.vit_mode = vit_mode

        all_folds = list(range(1, 9))
        fold_seq = {0: [1, 3],
                    1: [2, 5],
                    2: [4, 8],
                    3: [6, 7]}

        if mode == "train":
            seqs = [x for x in all_folds if x not in fold_seq[fold]]
        elif mode == "val":
            seqs = fold_seq[fold]

        self.mask_dir = osp.join(data_root_dir, str(version), "binary_annotations")

        self.mask_list = []
        for seq in seqs:
            seq_path = osp.join(self.mask_dir, f"seq{seq}")
            self.mask_list += [f"seq{seq}/{mask}" for mask in os.listdir(seq_path)]

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, index):
        mask_name = self.mask_list[index]

        # get class id from mask_name
        cls_id = int(re.search(r"class(\d+)", mask_name).group(1))

        # get pre-computed sam feature
        feat_dir = osp.join(self.mask_dir.replace("binary_annotations", f"sam_features_{self.vit_mode}"), mask_name.split("_")[0] + ".npy")
        sam_feat = np.load(feat_dir)

        # get ground-truth mask
        mask_path = osp.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # get class embedding
        class_embedding_path = osp.join(self.mask_dir.replace("binary_annotations", f"class_embeddings_{self.vit_mode}"), mask_name.replace("png","npy"))
        class_embedding = np.load(class_embedding_path)

        return sam_feat, mask_name, cls_id, mask, class_embedding

DATA_ROOT = "/mnt/hdd2/task2/surgicalsam/dataset"

class SurgicalInternalDataset(Dataset):
    """One (frame, class) row of the internal dataset.

    Expects a pandas DataFrame with columns:
        png_filename         e.g. /mnt/hdd2/task2/{train,val,test}_set/{pid}/C?-XXXXX.png
        label_id             int (1..28)
        binary_annotations   e.g. /mnt/hdd2/task2/surgicalsam/dataset/binary_annotations/{pid}/C?-XXXXX_classN.png
    """

    def __init__(self, df, data_root=DATA_ROOT):
        self.png = df["png_filename"].tolist()
        self.cls = df["label_id"].tolist()
        self.bm = df["binary_annotations"].tolist()
        self.data_root = data_root

    def __len__(self):
        return len(self.png)

    def __getitem__(self, idx):
        cls_id = int(self.cls[idx])
        bm_path = self.bm[idx]

        # SAM feature: map the original frame dir to surgicalsam/dataset/feat_dir
        dir_path, file_name = os.path.split(self.png[idx])
        feat_dir = dir_path
        for s in ("train_set", "val_set", "test_set"):
            feat_dir = feat_dir.replace(s, "surgicalsam/dataset/feat_dir")
        sam_feat = np.load(os.path.join(feat_dir, file_name.replace("png", "npy")))

        # per-class binary mask
        mask = cv2.imread(bm_path, cv2.IMREAD_GRAYSCALE)

        # mask name as "{pid}/{frame}_classN.png"  (used to rebuild the semantic map)
        pid_mask = bm_path.split("/")[-2] + "/" + bm_path.split("/")[-1]

        # class embedding
        ce_path = os.path.join(self.data_root, "class_embeddings", pid_mask.replace(".png", ".npy"))
        try:
            class_embedding = np.load(ce_path)
        except Exception as e:
            print(f"[warn] class embedding missing: {ce_path}: {e}")
            class_embedding = np.zeros((256,), dtype=np.float32)

        return sam_feat, pid_mask, cls_id, mask, class_embedding


def read_gt_masks_internal(nrrd_paths, H=1004):
    """Read each unique GT nrrd once -> {"{pid}/{frame}.png": (H, W) int tensor}.
       The nrrd is (W, H_full, 1); we keep the first H rows and transpose to (H, W)."""
    import nrrd
    gt = {}
    for p in dict.fromkeys(nrrd_paths):  # dedupe, preserve order
        key = p.split("/")[-2] + "/" + p.split("/")[-1].replace("-labels.nrrd", ".png")
        data, _ = nrrd.read(p)
        lab = data[:, :H, 0].squeeze().T
        gt[key] = torch.from_numpy(lab.astype(np.int32))
    return gt
