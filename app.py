import streamlit as st
import torch
import cv2
import numpy as np
from torchvision import transforms
import segmentation_models_pytorch as smp
from datagen import preprocessing, predict_mask, generate_pseudo_mask

# ------------------------------------------
# Classifier Imports
# ------------------------------------------
from torchvision import models
import torch.nn as nn

# ------------------------------------------
# Streamlit Page UI
# ------------------------------------------
st.set_page_config(page_title="Lung Tumor Segmentation + Classification", layout="wide")
st.title("🫁 Lung Tumor Segmentation + Tumor Classification")
st.write("Upload a chest CT/X-ray image to segment lungs/tumor and classify as benign or malignant.")

# ------------------------------------------
# Load U-Net Segmentation Model
# ------------------------------------------
@st.cache_resource
def load_seg_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load("lung_tumor_unet_resnet34.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, device

seg_model, seg_device = load_seg_model()

# ------------------------------------------
# Load DenseNet Classifier Model
# ------------------------------------------
@st.cache_resource
def load_classifier_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build DenseNet
    model = models.densenet121(weights=None)
    in_feat = model.classifier.in_features
    model.classifier = nn.Linear(in_feat, 2)  # 2 classes: benign/malignant

    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, device

classifier_model, classifier_device = load_classifier_model()
labels_map = {0: "Benign", 1: "Malignant"}

# ------------------------------------------
# File Upload
# ------------------------------------------
uploaded_file = st.file_uploader("Upload CT/X-ray Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    # ----------------------------
    # Read image
    # ----------------------------
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (256, 256))

    st.subheader("📌 Original Image")
    st.image(image_resized, width=256)

    # ----------------------------
    # Preprocessing
    # ----------------------------
    enhanced, lung_fields, lung_mask = preprocessing(image_resized)
    gen_mask = generate_pseudo_mask(lung_fields)
    final_mask = cv2.bitwise_and(gen_mask, gen_mask, mask=lung_mask)
    colored_mask = cv2.applyColorMap(final_mask, cv2.COLORMAP_JET)
    lung_fields = cv2.addWeighted(lung_fields, 0.7, colored_mask, 0.3, 0)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("✨ Enhanced Image")
        st.image(enhanced, width=256)
    with col2:
        st.subheader("🫁 Extracted Lung Fields")
        st.image(lung_fields, width=256)

    # ----------------------------
    # Segmentation Prediction
    # ---------------------------
    img_gray = cv2.cvtColor(lung_fields, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_tensor = transforms.ToTensor()(lung_fields).unsqueeze(0).to(seg_device)
    mask = predict_mask(seg_model, img_tensor, seg_device)

    cv2.drawContours(mask, contours, -1, color=255, thickness=-1)  # Fill the contours
    mask = cv2.bitwise_and(img_gray, img_gray, mask=mask)
    count = np.count_nonzero(mask)

    if mask.max() == 0 or count < 500:
        hsv = cv2.cvtColor(lung_fields, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([90, 50, 20])
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        mask = cv2.bitwise_and(lung_fields, lung_fields, mask=cv2.bitwise_not(mask))

    st.subheader("🔍 Predicted Tumor Mask")
    st.image(mask, clamp=True, width=256)

    # ----------------------------
    # Tumor Classification
    # ----------------------------
    # Extract tumor region using mask

    mask = mask.astype(np.uint8)

    if len(mask.shape) >2:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)



    print(mask.shape)
    print(lung_fields.shape)


    tumor_region = cv2.bitwise_and(lung_fields, lung_fields, mask=mask)

    # Resize & normalize for DenseNet
    tumor_tensor = cv2.resize(tumor_region, (224, 224))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tumor_tensor = transform(tumor_tensor).unsqueeze(0).to(classifier_device)

    tumor_area = np.count_nonzero(mask)
    print(tumor_area)
    threshold = 5000

    # Predict benign/malignant

    if tumor_area < threshold:
        pred_class = "Benign"
    else:

        with torch.no_grad():
            out = classifier_model(tumor_tensor)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            pred_label = int(np.argmax(probs))
            pred_class = labels_map[pred_label]

    st.subheader("🧬 Tumor Classification")
    st.write(f"Predicted Class: **{pred_class}**")

    st.image(tumor_region, caption="Tumor Region for Classification", width=256)
