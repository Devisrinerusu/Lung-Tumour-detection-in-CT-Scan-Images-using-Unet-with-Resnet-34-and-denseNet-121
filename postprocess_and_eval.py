# postprocess_and_eval.py
import os
import time
import cv2
import numpy as np
from skimage.morphology import remove_small_objects, disk, closing, opening
from skimage.measure import label, regionprops
from skimage.segmentation import chan_vese
from skimage.metrics import hausdorff_distance
from scipy.ndimage import binary_fill_holes, gaussian_filter

def keep_largest_cc(binary_mask):
    """
    Keep only the largest connected component in `binary_mask`.
    Input: binary_mask (uint8 0/255) or bool
    Output: binary mask (uint8 0/255)
    """
    bin_bool = (binary_mask > 0)
    labeled = label(bin_bool)
    regions = regionprops(labeled)
    if not regions:
        return np.zeros_like(binary_mask, dtype=np.uint8)
    largest = max(regions, key=lambda r: r.area)
    out = (labeled == largest.label).astype(np.uint8) * 255
    return out

def morph_smooth(binary_mask, closing_radius=5, min_size=500):
    """
    Smooth mask with morphological closing, remove small objects and fill holes.
    """
    mask_bool = binary_mask > 0
    # closing
    closed = closing(mask_bool, disk(closing_radius))
    opened = opening(closed, disk(3))
    # remove small objects
    cleaned = remove_small_objects(opened, min_size=min_size)
    # fill holes
    filled = binary_fill_holes(cleaned)
    return (filled.astype(np.uint8) * 255)

def active_contour_refine(gray_image, init_mask, iters=80):
    """
    Use skimage's Chan-Vese active contour to refine the initial mask.
    gray_image: grayscale image (0-255)
    init_mask: binary mask (0/255) resized to same shape
    """
    try:
        if init_mask.dtype != bool:
            init_level = init_mask > 0
        else:
            init_level = init_mask
        # normalize image to [0,1] float
        img = (gray_image.astype(np.float32) - gray_image.min()) / (gray_image.max() - gray_image.min() + 1e-8)
        cv_result = chan_vese(img, mu=0.25, init_level_set=init_level, max_num_iter=iters, extended_output=False)
        return (cv_result.astype(np.uint8) * 255)
    except Exception as e:
        print("Chan-Vese failed:", e)
        return init_mask

# Pixel-level metrics
def dice_coef(gt, pred, eps=1e-7):
    gt_bin = (gt > 0).astype(np.uint8)
    pr_bin = (pred > 0).astype(np.uint8)
    inter = (gt_bin & pr_bin).sum()
    return (2.0 * inter + eps) / (gt_bin.sum() + pr_bin.sum() + eps)

def iou_score(gt, pred, eps=1e-7):
    gt_bin = (gt > 0).astype(np.uint8)
    pr_bin = (pred > 0).astype(np.uint8)
    inter = (gt_bin & pr_bin).sum()
    uni = (gt_bin | pr_bin).sum()
    return (inter + eps) / (uni + eps)

def sensitivity_specificity(gt, pred):
    gt_bin = (gt > 0).astype(np.uint8)
    pr_bin = (pred > 0).astype(np.uint8)
    TP = np.logical_and(gt_bin == 1, pr_bin == 1).sum()
    TN = np.logical_and(gt_bin == 0, pr_bin == 0).sum()
    FP = np.logical_and(gt_bin == 0, pr_bin == 1).sum()
    FN = np.logical_and(gt_bin == 1, pr_bin == 0).sum()
    sens = TP / (TP + FN + 1e-7)
    spec = TN / (TN + FP + 1e-7)
    return sens, spec

def compute_hausdorff(gt, pred):
    gt_bin = (gt > 0).astype(np.uint8)
    pr_bin = (pred > 0).astype(np.uint8)
    # if either is empty, define large distance
    if gt_bin.sum() == 0 or pr_bin.sum() == 0:
        return np.nan
    try:
        return hausdorff_distance(gt_bin, pr_bin)
    except Exception:
        return np.nan

def postprocess_pipeline(raw_mask, gray_image):
    """
    Input:
      raw_mask: float mask predicted by NN in 0/255 or 0/1 or float
      gray_image: corresponding grayscale image for active contour refinement
    Returns: refined_mask (0/255 uint8)
    Steps:
      - threshold raw_mask
      - morphological smooth
      - keep largest connected comp
      - active contour refine
      - final small-object removal
    """
    # ensure binary
    if raw_mask.dtype == np.float32 or raw_mask.dtype == np.float64:
        raw_mask = (raw_mask * 255.0).astype(np.uint8) if raw_mask.max() <= 1.0 else raw_mask.astype(np.uint8)
    thresh = (raw_mask > 127).astype(np.uint8) * 255

    sm = morph_smooth(thresh, closing_radius=6, min_size=300)
    largest = keep_largest_cc(sm)
    refined = active_contour_refine(gray_image, largest, iters=80)
    refined = morph_smooth(refined, closing_radius=4, min_size=200)
    return refined

def extract_roi_bbox(refined_mask, pad=10, keep_square=False, square_size=None):
    """
    Return bounding box [x1,y1,x2,y2] of refined_mask (binary 0/255).
    Optionally add padding.
    """
    ys, xs = np.where(refined_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = max(xs.min() - pad, 0), min(xs.max() + pad, refined_mask.shape[1] - 1)
    y1, y2 = max(ys.min() - pad, 0), min(ys.max() + pad, refined_mask.shape[0] - 1)
    if keep_square:
        h = y2 - y1
        w = x2 - x1
        s = square_size if square_size is not None else max(h, w)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        half = s // 2
        x1 = max(cx - half, 0); x2 = min(cx + half, refined_mask.shape[1] - 1)
        y1 = max(cy - half, 0); y2 = min(cy + half, refined_mask.shape[0] - 1)
    return [x1, y1, x2, y2]

# Convenience: crop roi safely
def crop_roi(image, bbox):
    if bbox is None:
        return None
    x1,y1,x2,y2 = bbox
    return image[y1:y2+1, x1:x2+1]

# evaluation helper to run on lists
def evaluate_dataset(gt_list, pred_list):
    """
    gt_list, pred_list: equal-length lists of (H,W) uint8 masks (0/255)
    returns dict of aggregated scores (mean) and lists
    """
    assert len(gt_list) == len(pred_list)
    dices=[]; ious=[]; sens=[]; spec=[]; haus=[]

    for gt, pr in zip(gt_list, pred_list):
        dices.append(dice_coef(gt, pr))
        ious.append(iou_score(gt, pr))
        s, sp = sensitivity_specificity(gt, pr)
        sens.append(s); spec.append(sp)
        haus.append(compute_hausdorff(gt, pr))

    import numpy as np
    stats = {
        "dice_mean": float(np.nanmean(dices)),
        "iou_mean": float(np.nanmean(ious)),
        "sensitivity_mean": float(np.nanmean(sens)),
        "specificity_mean": float(np.nanmean(spec)),
        "hausdorff_median": float(np.nanmedian([h for h in haus if not np.isnan(h)])) if any([not np.isnan(h) for h in haus]) else float('nan'),
        "num_samples": len(dices)
    }
    return stats, {"dice": dices, "iou": ious, "sensitivity": sens, "specificity": spec, "hausdorff": haus}
