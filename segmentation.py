import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import cv2
import numpy as np

# ---------------------- Custom Dataset Class ----------------------
class LungDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):  # image,mask paths
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):   ## get item
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   #converting color from bgr to rgb
        img = cv2.resize(img, (256, 256))

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE) # reading masks in black and white
        mask = cv2.resize(mask, (256, 256))
        mask = np.expand_dims(mask, axis=-1)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = transforms.ToTensor()(img)
        mask = torch.from_numpy(mask.transpose((2, 0, 1))).float() / 255.0 # normalization the iamges
        return img, mask

# ---------------------- U-Net Model Definition ----------------------
def get_unet_model():
    model = smp.Unet(
        encoder_name="resnet34",        # Pretrained on ImageNet
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,                      # Binary mask

    )
    return model

# ---------------------- Training Setup ----------------------
def train_unet(model, train_loader, val_loader, epochs=5, lr=1e-4, device='cuda'):
    criterion = smp.losses.DiceLoss(mode='binary')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model = model.to(device)

    for epoch in range(epochs):
        print(f"\n====== Epoch {epoch + 1}/{epochs} ======")

        model.train()
        train_loss = 0

        for batch_idx, (images, masks) in enumerate(tqdm(train_loader)):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # print batch loss every 10 steps
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx + 1}/{len(train_loader)} Loss: {loss.item():.4f}")

        # ---------- Validation ----------
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        print(f"Train Loss: {train_loss / len(train_loader):.4f}")
        print(f"Valid Loss: {val_loss / len(val_loader):.4f}")

    torch.save(model.state_dict(), "lung_tumor_unet_resnet34.pth")
    print("Training done. Model saved!")
