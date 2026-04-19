import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from sklearn.model_selection import train_test_split
from segmentation import LungDataset, train_unet, get_unet_model
import torch
from torch.utils.data import DataLoader
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from torchvision import transforms



def preprocessing(image, clip_limit=2.0, grid_size=(8, 8)):
    # ---------- Step 1: Gaussian Blur ----------
    img_blur = cv2.GaussianBlur(image, (5, 5), 0)

    # ---------- Step 2: Sobel edge enhancement ----------
    sobelx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=5)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = cv2.convertScaleAbs(sobel)

    # ---------- Step 3: CLAHE ----------
    lab = cv2.cvtColor(img_blur, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    img_clahe = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    # ---------- Step 4: Enhancement (CLAHE + Sobel) ----------
    enhanced = cv2.addWeighted(img_clahe, 0.8, sobel, 0.2, 0)

    # ---------- Step 5: Lung Field Extraction ----------
    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]

    lung_mask = np.zeros_like(gray)
    cv2.drawContours(lung_mask, contours, -1, 200, -1)

    lung_fields = cv2.bitwise_and(enhanced, enhanced, mask=lung_mask)

    # # ---------- Visualization ----------
    # plt.figure(figsize=(15, 5))
    #
    # plt.subplot(1, 5, 1)
    # plt.imshow(image)
    # plt.axis('off')
    # plt.title('Original', fontname='Times New Roman', fontweight='bold')
    #
    # plt.subplot(1, 5, 2)
    # plt.imshow(img_blur)
    # plt.axis('off')
    # plt.title('Blurred', fontname='Times New Roman', fontweight='bold')
    #
    # plt.subplot(1, 5, 3)
    # plt.imshow(img_clahe)
    # plt.axis('off')
    # plt.title('CLAHE', fontname='Times New Roman', fontweight='bold')
    #
    # plt.subplot(1, 5, 4)
    # plt.imshow(enhanced)
    # plt.axis('off')
    # plt.title('Enhanced', fontname='Times New Roman', fontweight='bold')
    #
    # plt.subplot(1, 5, 5)
    # plt.imshow(lung_fields)
    # plt.axis('off')
    # plt.title('Lung Fields', fontname='Times New Roman', fontweight='bold')
    #
    # plt.tight_layout()
    # plt.savefig("Data Visualization/preprocessed_lung_fields.png")
    # plt.show()

    return enhanced, lung_fields, lung_mask


def generate_pseudo_mask(lung_field_img):
    # Convert to grayscale
    gray = cv2.cvtColor(lung_field_img, cv2.COLOR_RGB2GRAY)

    # Thresholding to isolate potential nodules
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleaning
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Keep only largest connected component (assumed to be tumor region)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closing, connectivity=8)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        pseudo_mask = np.uint8(labels == largest) * 255
    else:
        pseudo_mask = closing

    return pseudo_mask

def datagen():
    # train_path = 'dataset 1/LungcancerDataSet/Data/train'
    train_path = 'dataset 2/The IQ-OTHNCCD lung cancer dataset'
    labels_fold = os.listdir(train_path)

    Labels = []
    for lab in labels_fold:
        images = os.listdir(train_path+f'/{lab}')
        for img in images:
            image = cv2.imread(f'{train_path}/{lab}/{img}')
            image = cv2.resize(image, (256, 256))

            enhanced_img, lung_field, _ = preprocessing(image)
            mask = generate_pseudo_mask(lung_field)
            os.makedirs(f'dataset_auto_masks/dataset 2/{lab}', exist_ok=True)
            cv2.imwrite(f"dataset_auto_masks/dataset 2/{lab}/{img}", mask)
            Labels.append(lab)
            x = 0


image_paths = []
mask_paths = []

def train_for_seg():
    def seg(paths):

        image_root = paths[0]
        mask_root = paths[1]

        classes = os.listdir(image_root)

        for cls in classes:
            img_files = glob.glob(f"{image_root}/{cls}/*")
            for img_path in img_files:
                img_name = os.path.basename(img_path)

                mask_path = f"{mask_root}/{cls}/{img_name}"

                if os.path.exists(mask_path):  # ensure mask exists
                    image_paths.append(img_path)
                    mask_paths.append(mask_path)
                else:
                    print("❌ Mask missing for:", img_path)
        x = 0

    pathss = [("dataset 2/The IQ-OTHNCCD lung cancer dataset","dataset_auto_masks/dataset 2"), ("dataset 1/LungcancerDataSet/Data/train", "dataset_auto_masks/dataset 1")]

    for path in pathss:
        seg(path)

    train_images, val_images, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )

    train_dataset = LungDataset(train_images, train_masks)
    val_dataset = LungDataset(val_images, val_masks)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


    model = get_unet_model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trained_model = train_unet(
        model,
        train_loader,
        val_loader,
        epochs=25,
        lr=1e-4,
        device=device
    )

    x = 0


def load_unet_model(weights_path, device='cuda'):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,   # <-- important (don’t load imagenet now)
        in_channels=3,
        classes=1
    )

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def predict_mask(model, img_tensor, device='cuda'):
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = model(img_tensor)

    mask = torch.sigmoid(output).cpu().numpy()[0, 0]  # shape → (256, 256)
    mask = (mask > 0.5).astype(np.uint8) * 255         # binary mask

    return mask

def save_mask(mask, save_path):
    cv2.imwrite(save_path, mask)

def segmentation():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_unet_model("lung_tumor_unet_resnet34.pth", device)

    image_path = "dataset 1/LungcancerDataSet/Data/train/adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib/000005 (9).png"  # <-- your image
    save_path = "predicted_mask.png"  # output mask file

    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))

    enhanced, lung_fields = preprocessing(image)

    img = cv2.resize(enhanced, (256, 256))

    img_tensor = transforms.ToTensor()(img).unsqueeze(0)

    mask = predict_mask(model, img_tensor, device)
    save_mask(mask, save_path)

    print("Segmentation completed. Mask saved at:", save_path)
