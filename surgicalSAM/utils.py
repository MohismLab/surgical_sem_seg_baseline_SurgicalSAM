import numpy as np 
import cv2 
import torch 
import os 
import os.path as osp 
import re 

def create_binary_masks(binary_masks, preds, preds_quality, mask_names, thr):

    """Gather the predicted binary masks of different frames and classes into a dictionary, mask quality is also recorded

    Returns:
        dict: a dictionary containing all predicted binary masks organised based on sequence, frame, and mask name
    """
    preds = preds.cpu()
    preds_quality = preds_quality.cpu()
    
    pred_masks = (preds > thr).int()
    

    for pred_mask, mask_name, pred_quality in zip(pred_masks, mask_names, preds_quality):        
      
        seq_name = mask_name.split("/")[0]
        frame_name = osp.basename(mask_name).split("_")[0]

        # QL modify "保存pred_mask为图片"
        # from PIL import Image
        # save_dir = r"./pred_masks"
        # pred_mask_img = (pred_mask.numpy() * 255).astype(np.uint8)
        # pred_mask_img = np.squeeze(pred_mask_img)
        # pred_mask_img_path = os.path.join(save_dir, f"{seq_name}_{frame_name}_pred_mask.png")
        # Image.fromarray(pred_mask_img).save(pred_mask_img_path)
        
        if seq_name not in binary_masks.keys():
            binary_masks[seq_name] = dict()
        
        if frame_name not in binary_masks[seq_name].keys():
            binary_masks[seq_name][frame_name] = list()
            
        binary_masks[seq_name][frame_name].append({
            "mask_name": mask_name,
            "mask": pred_mask,
            "mask_quality": pred_quality.item()
        })
        
    return binary_masks
        

def create_endovis_masks(binary_masks, H, W):
    """given the dictionary containing all predicted binary masks, compute final prediction of each frame and organise the prediction masks into a dictionary
       H - height of image 
       W - width of image
    
    Returns: a dictionary containing one prediction mask for each frame with the frame name as key and its predicted mask as value; 
             For each frame, the binary masks of different classes are conbined into a single prediction mask;
             The prediction mask for each frame is a 1024 x 1280 map with each value representing the class id for the pixel;
             
    """
    
    endovis_masks = dict()
    
    for seq in binary_masks.keys():
        
        for frame in binary_masks[seq].keys():
            
            endovis_mask = np.zeros((H, W))
    
            binary_masks_list = binary_masks[seq][frame]

            binary_masks_list = sorted(binary_masks_list, key=lambda x: x["mask_quality"])
           
            for binary_mask in binary_masks_list:
                # '10/C2-00000_class15.png'
                mask_name  = binary_mask["mask_name"]

                predicted_label = int(re.search(r"class(\d+)", mask_name).group(1))

                # print(f"Predicted label: {predicted_label}")

                # mask.shape = (1024, 1280)
                mask = binary_mask["mask"].numpy()

                # print(f"mask.sum(): {mask.sum()}, {np.unique(mask)}")

                # endovis_mask.shape = (1004, 1920)
                endovis_mask[mask==1] = predicted_label

                # print(f"endovis_mask1: {endovis_mask.sum()/mask.sum()}, {np.unique(endovis_mask)}")

            # print(f"endovis_mask shape af loop: {endovis_mask.shape}, {np.unique(endovis_mask)}")

            endovis_mask = endovis_mask.astype(int)
            
            # print(f"endovis_mask shape2: {endovis_mask.shape}, {np.unique(endovis_mask)}")

            endovis_masks[f"{seq}/{frame}.png"] = endovis_mask

            # print(f"{seq}/{frame}.png_endovis_masks3: {np.unique(endovis_masks['10/C2-00000.png'])}")
    
    # print(f"endovis_masks4: {np.unique(endovis_masks['10/C2-00000.png'])}")
    return endovis_masks


def eval_endovis(endovis_masks, gt_endovis_masks):
    """Given the predicted masks and groundtruth annotations, predict the challenge IoU, IoU, mean class IoU, and the IoU for each class
        
      ** The evaluation code is taken from the official evaluation code of paper: ISINet: An Instance-Based Approach for Surgical Instrument Segmentation
      ** at https://github.com/BCV-Uniandes/ISINet
      
    Args:
        endovis_masks (dict): the dictionary containing the predicted mask for each frame 
        gt_endovis_masks (dict): the dictionary containing the groundtruth mask for each frame 

    Returns:
        dict: a dictionary containing the evaluation results for different metrics 
    """

    endovis_results = dict()
    num_classes = 28 # QL modify num_classes = 7
    
    all_im_iou_acc = []
    all_im_iou_acc_challenge = []
    cum_I, cum_U = 0, 0
    class_ious = {c: [] for c in range(1, num_classes+1)}
    
    for file_name, prediction in endovis_masks.items():
       
        full_mask = gt_endovis_masks[file_name]
        
        im_iou = []
        im_iou_challenge = []
        target = full_mask.numpy()
        gt_classes = np.unique(target)
        gt_classes.sort()
        gt_classes = gt_classes[(gt_classes > 0) & (gt_classes != 29)] # 排除背景类和排除label 29（不参与此次任务）       
        # QL modify gt_classes = gt_classes[gt_classes > 0] 
        if np.sum(prediction) == 0:
            if target.sum() > 0: 
                all_im_iou_acc.append(0)
                all_im_iou_acc_challenge.append(0)
                for class_id in gt_classes:
                    class_ious[class_id].append(0)
            continue

        gt_classes = torch.unique(full_mask)
        gt_classes = gt_classes[(gt_classes > 0) & (gt_classes != 29)] # 排除背景类和排除label 29（不参与此次任务）       
        # loop through all classes from 1 to num_classes 
        for class_id in gt_classes: # QL modify range(1, num_classes + 1)
            class_id = class_id.item()  # QL modify 转为Python整数
            current_pred = (prediction == class_id).astype(np.float64)
            current_target = (full_mask.numpy() == class_id).astype(np.float64)

            if current_pred.astype(np.float64).sum() != 0 or current_target.astype(np.float64).sum() != 0:
                # compute_mask_IU_endovis(masks, target)
                i, u = compute_mask_IU_endovis(current_pred, current_target)     
                if u == 0:
                    iou = 0.0
                else:
                    iou = i / u
                im_iou.append(iou)
                cum_I += i
                cum_U += u
                class_ious[class_id].append(iou)
                if class_id in gt_classes:
                    im_iou_challenge.append(iou)
        
        if len(im_iou) > 0:
            all_im_iou_acc.append(np.mean(im_iou))
        if len(im_iou_challenge) > 0:
            all_im_iou_acc_challenge.append(np.mean(im_iou_challenge))

    # calculate final metrics
    final_im_iou = cum_I / (cum_U + 1e-15)
    mean_im_iou = np.mean(all_im_iou_acc)
    mean_im_iou_challenge = np.mean(all_im_iou_acc_challenge)

    final_class_im_iou = torch.zeros(num_classes)
    cIoU_per_class = []
    for c in gt_classes: # QL modify range(1, num_classes + 1)
        c = c.item()
        final_class_im_iou[c-1] = torch.tensor(class_ious[c]).float().mean()
        cIoU_per_class.append(round((final_class_im_iou[c-1]*100).item(), 3))
        
    mean_class_iou = torch.tensor([torch.tensor(values).float().mean() for c, values in class_ious.items() if len(values) > 0]).mean().item()
    
    endovis_results["challengIoU"] = round(mean_im_iou_challenge*100,3)
    endovis_results["IoU"] = round(mean_im_iou*100,3)
    endovis_results["mcIoU"] = round(mean_class_iou*100,3)
    endovis_results["mIoU"] = round(final_im_iou*100,3)
    
    endovis_results["cIoU_per_class"] = cIoU_per_class
    
    return endovis_results



def compute_mask_IU_endovis(masks, target):
    """compute iou used for evaluation
    """
    # target.shape = (1004, 1920) masks.shape = (1004, 1920)
    assert target.shape[-2:] == masks.shape[-2:]
    temp = masks * target
    intersection = temp.sum()
    union = ((masks + target) - temp).sum()
    return intersection, union


def read_gt_endovis_masks(data_root_dir = "../data/endovis_2018",
                          mode = "val", 
                          fold = None):
    
    """Read the annotation masks into a dictionary to be used as ground truth in evaluation.

    Returns:
        dict: mask names as key and annotation masks as value 
    """
    gt_endovis_masks = dict()
    
    if "2018" in data_root_dir:
        gt_endovis_masks_path = osp.join(data_root_dir, mode, "annotations")
        for seq in os.listdir(gt_endovis_masks_path):
            for mask_name in os.listdir(osp.join(gt_endovis_masks_path, seq)):
                full_mask_name = f"{seq}/{mask_name}"
                mask = torch.from_numpy(cv2.imread(osp.join(gt_endovis_masks_path, full_mask_name),cv2.IMREAD_GRAYSCALE))
                gt_endovis_masks[full_mask_name] = mask
                
    elif "2017" in data_root_dir:
        if fold == "all":
            seqs = [1,2,3,4,5,6,7,8]
            
        elif fold in [0,1,2,3]:
            fold_seq = {0: [1, 3],
                        1: [2, 5],
                        2: [4, 8],
                        3: [6, 7]}
            
            seqs = fold_seq[fold]
        
        gt_endovis_masks_path = osp.join(data_root_dir, "0", "annotations")
        
        for seq in seqs:
            for mask_name in os.listdir(osp.join(gt_endovis_masks_path, f"seq{seq}")):
                full_mask_name = f"seq{seq}/{mask_name}"
                mask = torch.from_numpy(cv2.imread(osp.join(gt_endovis_masks_path, full_mask_name),cv2.IMREAD_GRAYSCALE))
                gt_endovis_masks[full_mask_name] = mask
            
            
    return gt_endovis_masks


def print_log(str_to_print, log_file):
    """Print a string and meanwhile write it to a log file
    """
    print(str_to_print)
    with open(log_file, "a") as file:
        file.write(str_to_print+"\n")


class EndovisMaskAccumulator:

    def __init__(self, H, W):
        self.H, self.W = H, W
        self.sem = {}     # "{seq}/{frame}.png" -> uint8 (H, W) semantic map
        self.qual = {}    # "{seq}/{frame}.png" -> float32 (H, W) best quality

    def add(self, preds, preds_quality, mask_names, thr=0):
        masks = (preds.detach().cpu() > thr).numpy()        # (B, H, W) bool
        quals = preds_quality.detach().cpu().reshape(-1)    # (B,)
        for m, mask_name, q in zip(masks, mask_names, quals):
            seq = mask_name.split("/")[0]
            frame = osp.basename(mask_name).split("_")[0]
            key = f"{seq}/{frame}.png"
            cls = int(re.search(r"class(\d+)", mask_name).group(1))
            if key not in self.sem:
                self.sem[key] = np.zeros((self.H, self.W), dtype=np.uint8)
                self.qual[key] = np.full((self.H, self.W), -np.inf, dtype=np.float32)
            qv = float(q)
            better = m & (qv >= self.qual[key])
            self.sem[key][better] = cls
            self.qual[key][better] = qv

    def result(self):
        """Return {frame_key: (H, W) uint8 semantic map}.

        Values are class ids 0..28 (uint8 represents them exactly), so this is
        byte-identical in content to the previous astype(int) version while using
        1/8 the memory. We also drop the float32 quality buffers here: once the
        semantic maps are final, quality is no longer needed, so freeing it cuts
        the validation memory peak from ~78GB to ~25GB (no OOM).
        """
        self.qual.clear()
        return self.sem
