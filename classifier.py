# classifier.py
import os
import time
import cv2
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from torchvision.transforms import InterpolationMode
from postprocess_and_eval import extract_roi_bbox, crop_roi

import torch
import cv2
import numpy as np
from torchvision import transforms

from datagen import preprocessing, load_unet_model, predict_mask, generate_pseudo_mask

# Simple dataset that returns cropped patch + label
class TumorPatchDataset(Dataset):
    def __init__(self, image_paths, labels,
                 model, device="cuda",
                 transform=None,
                 patch_size=224,
                 save_mask_root=None):

        self.image_paths = image_paths
        self.labels = labels #---benign or maligant
        self.model = model #--U-Net Model
        self.device = device
        self.transform = transform
        self.patch_size = patch_size # segmented images input to classification that is DenseNet-121
        self.save_mask_root = save_mask_root

        if save_mask_root:
            os.makedirs(save_mask_root, exist_ok=True)

    # ------------------------------------------------------------------
    # 🎯 EXACT segmentation pipeline from your Streamlit app
    # ------------------------------------------------------------------
    def generate_mask(self, img_rgb, save_path=None):

        # ----------------------------
        # Preprocessing
        # ----------------------------
        enhanced, lung_fields, lung_mask = preprocessing(img_rgb)
        gen_mask = generate_pseudo_mask(lung_fields)

        final_mask = cv2.bitwise_and(gen_mask, gen_mask, mask=lung_mask)
        colored_mask = cv2.applyColorMap(final_mask, cv2.COLORMAP_JET)
        lung_fields = cv2.addWeighted(lung_fields, 0.7, colored_mask, 0.3, 0)

        # ----------------------------
        # U-Net prediction
        # ----------------------------
        gray = cv2.cvtColor(lung_fields, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_tensor = transforms.ToTensor()(lung_fields).unsqueeze(0).to(self.device)
        mask = predict_mask(self.model, img_tensor, self.device)

        cv2.drawContours(mask, contours, -1, color=255, thickness=-1)
        mask = cv2.bitwise_and(gray, gray, mask=mask) #--- combine masks

        # fallback
        if mask.max() == 0 or np.count_nonzero(mask) < 500:
            hsv = cv2.cvtColor(lung_fields, cv2.COLOR_RGB2HSV)
            lower_blue = np.array([90, 50, 20])
            upper_blue = np.array([140, 255, 255])
            fallback = cv2.inRange(hsv, lower_blue, upper_blue)
            mask = cv2.bitwise_not(fallback)

        if save_path:
            cv2.imwrite(save_path, mask)

        return mask

    # ------------------------------------------------------------------
    # Dataset __getitem__
    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        path = self.image_paths[idx]

        img = cv2.imread(path)                        #--load the images
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))

        # mask save path
        if self.save_mask_root:
            cls = os.path.basename(os.path.dirname(path))
            fname = os.path.basename(path)
            save_dir = os.path.join(self.save_mask_root, cls)
            os.makedirs(save_dir, exist_ok=True)
            mask_save_path = os.path.join(save_dir, fname)
        else:
            mask_save_path = None

        # generate the mask (Streamlit logic)
        mask = self.generate_mask(img, save_path=mask_save_path)    #-- generate refined masks

        # ROI crop
        bbox = extract_roi_bbox(mask, pad=10, keep_square=True, square_size=self.patch_size) #-- Extract ROI and find tumor
        patch = crop_roi(img, bbox)

        if patch is None:  # if tumor is none then and handle ROI using full iamge
            patch = cv2.resize(img, (self.patch_size, self.patch_size))
        else:
            patch = cv2.resize(patch, (self.patch_size, self.patch_size))

        if self.transform:
            patch = self.transform(patch)
        else:
            tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.patch_size, self.patch_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
            patch = tf(patch)

        return patch, torch.tensor(self.labels[idx], dtype=torch.long)

    def __len__(self):
        return len(self.image_paths)

#   BUILDING DenseNet 121 Model
def build_densenet(num_classes=2, pretrained=True):
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
    in_feat = model.classifier.in_features
    model.classifier = nn.Linear(in_feat, num_classes)
    return model
#Training the classifier
def train_classifier(model, train_loader, val_loader, device='cuda', epochs=10, lr=1e-4):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)

    best_val_acc = 0.0
    best_val_loss = float("inf")   # FIXED: initialize loss

    for epoch in range(epochs):
        model.train()
        total = 0; correct = 0; loss_acc = 0.0

        # ------------------- TRAINING -------------------
        for imgs, labs in train_loader:
            imgs = imgs.to(device); labs = labs.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labs)
            loss.backward()    #does backpropogation
            optimizer.step()

            loss_acc += loss.item()  # caluclates the loss
            preds = out.argmax(dim=1) # accurate caluclation
            total += labs.size(0)
            correct += (preds == labs).sum().item()

        train_acc = correct / total

        # ------------------- VALIDATION -------------------
        model.eval()
        total = 0; correct = 0; vloss = 0.0

        with torch.no_grad():
            for imgs, labs in val_loader:
                imgs = imgs.to(device); labs = labs.to(device)
                out = model(imgs)
                loss = criterion(out, labs)
                vloss += loss.item()

                preds = out.argmax(dim=1)
                total += labs.size(0)
                correct += (preds == labs).sum().item()

        val_acc = correct / total
        avg_val_loss = vloss / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} "
              f"train_loss:{loss_acc/len(train_loader):.4f} train_acc:{train_acc:.4f} "
              f"val_loss:{avg_val_loss:.4f} val_acc:{val_acc:.4f}")

        # ------------------- SAVE BEST MODEL -------------------
        if val_acc > best_val_acc or (val_acc == best_val_acc and avg_val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model (accuracy or loss improved)")

    return model

# loads the saved best classifier
def load_classifier(weights_path, device='cuda'):
    model = build_densenet(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model
#used for inferation
def infer_classifier(model, patch, device='cuda'):
    # patch is already preprocessed tensor C,H,W
    model = model.to(device)
    patch = patch.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(patch)
        pred = torch.softmax(out, dim=1).cpu().numpy()[0]
    label = int(np.argmax(pred))
    return label, pred
