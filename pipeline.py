# run_pipeline.py
import os
import cv2
import glob
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from postprocess_and_eval import postprocess_pipeline, extract_roi_bbox, crop_roi, evaluate_dataset
from classifier import build_densenet, train_classifier, load_classifier, TumorPatchDataset
from datagen import load_unet_model, predict_mask
from postprocess_and_eval import keep_largest_cc


import segmentation_models_pytorch as smp
import torch

def load_segmentation_model(weights_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )

    # FORCE CPU MAP LOCATION IF CUDA IS NOT AVAILABLE
    state_dict = torch.load(weights_path, map_location="cpu")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def compute_severity_score(refined_mask, lung_mask=None):
    """
    Compute severity score based on tumor% of lung area.
    refined_mask = final tumor mask (0/255)
    lung_mask = lung segmentation mask (optional)
    """
    # Convert masks to binary
    tumor_bin = (refined_mask > 0).astype(np.uint8)

    tumor_area = tumor_bin.sum()

    # lung area if lung mask provided, else use whole image
    if lung_mask is not None:
        lung_bin = (lung_mask > 0).astype(np.uint8)
        lung_area = lung_bin.sum()
    else:
        lung_area = refined_mask.size

    if lung_area == 0:
        return 0  # fall back to normal

    tumor_percent = (tumor_area / lung_area) * 100

    # convert % → severity class
    if tumor_percent < 1:
        severity = 0   # Normal
    elif tumor_percent < 5:
        severity = 1   # Mild
    elif tumor_percent < 15:
        severity = 2   # Moderate
    else:
        severity = 3   # Severe

    return severity


def run_on_folder(image_folder, gt_mask_folder=None, out_folder="Results", model_weights="lung_tumor_unet_resnet34.pth", classifier_weights=None):
    os.makedirs(out_folder, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load unet model
    unet = load_unet_model(model_weights, device)

    gt_masks = []
    pred_masks = []
    times = []

    img_paths = sorted(glob.glob(os.path.join(image_folder, "*", "*.*")))  # images inside class subfolders
    for img_path in img_paths:
        img = cv2.imread(img_path)
        orig_shape = img.shape[:2]

        try:
            from datagen import preprocessing  # your function defined earlier in datagen if present
            enhanced, lung_fields, _ = preprocessing(img)
            use_img = enhanced  # pass enhanced to model
        except Exception:
            # fallback: simple resize + rgb conversion
            use_img = cv2.resize(img, (256,256))
            use_img = cv2.cvtColor(use_img, cv2.COLOR_BGR2RGB)

        # prepare tensor
        from torchvision import transforms
        img_tensor = transforms.ToTensor()(cv2.resize(use_img, (256,256))).unsqueeze(0)
        start = time.time()
        raw_mask = predict_mask(unet, img_tensor, device)  # returns (256,256) binary 0/255
        elapsed = time.time()-start
        times.append(elapsed)

        # resize to original
        raw_mask_full = cv2.resize(raw_mask, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)

        # postprocess
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        refined = postprocess_pipeline(raw_mask_full, gray)

        # save results
        base = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(os.path.join(out_folder, base + "_raw_mask.png"), raw_mask_full)
        cv2.imwrite(os.path.join(out_folder, base + "_refined_mask.png"), refined)

        # collect for evaluation if ground truth available
        if gt_mask_folder:
            # match path
            # assume gt masks follow same relative structure and name
            rel = os.path.relpath(img_path, image_folder)
            gt_path = os.path.join(gt_mask_folder, rel)
            if os.path.exists(gt_path):
                gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                gt_res = cv2.resize(gt, (refined.shape[1], refined.shape[0]))
                gt_masks.append(gt_res)
                pred_masks.append(refined)
    # evaluation
    stats = None
    details = None
    if len(gt_masks) > 0:
        stats, details = evaluate_dataset(gt_masks, pred_masks)
    avg_time = float(np.mean(times)) if times else 0.0

    if stats:
        print("Evaluation stats:", stats)

    return stats, details, times

def normalize_class_name(cls):
    cls = cls.lower()
    if "bengin" in cls or "benign" in cls:
        return "benign"
    if "malignant" in cls:
        return "malignant"
    return None

def prepare_classifier_dataset(image_root, labels_map, seg_model, device="cuda",
                               split=0.2, batch_size=8, save_mask_root="generated_masks"):

    image_paths = []
    labels = []

    classes = os.listdir(image_root)
    classes = [c for c in classes if "Bengin" in c or "Malignant" in c]

    for cls in classes:
        cls_dir = os.path.join(image_root, cls)
        if not os.path.isdir(cls_dir):
            continue

        normalized = normalize_class_name(cls)
        label = labels_map[normalized]

        for img in os.listdir(cls_dir):
            image_paths.append(os.path.join(cls_dir, img))
            labels.append(label)

    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels, test_size=split, random_state=42, stratify=labels
    )

    train_ds = TumorPatchDataset(X_train, y_train, seg_model,
                                 device=device, save_mask_root=save_mask_root)

    val_ds = TumorPatchDataset(X_val, y_val, seg_model,
                               device=device, save_mask_root=save_mask_root)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


if __name__ == "__main__":

    # stats, details, times = run_on_folder("dataset 1/LungcancerDataSet/Data/train", gt_mask_folder="ground_truth_masks", out_folder="Results")
    from classifier import build_densenet, train_classifier

    seg_model = load_segmentation_model("lung_tumor_unet_resnet34.pth")

    labels_map = {"benign": 0, "malignant": 1}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader = prepare_classifier_dataset(
        image_root="dataset 1/LungcancerDataSet/Data/train",
        labels_map=labels_map,
        seg_model=seg_model,
        device=device,
        save_mask_root="generated_masks"
    )

    model = build_densenet(pretrained=True)
    model = train_classifier(model, train_loader, val_loader, device=device, epochs=100, lr=1e-4)
    x = 0