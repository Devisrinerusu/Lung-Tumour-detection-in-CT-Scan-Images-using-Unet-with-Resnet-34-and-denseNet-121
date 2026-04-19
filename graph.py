import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from matplotlib import rcParams

# ===============================
# GLOBAL FONT STYLE
# ===============================
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.weight'] = 'bold'
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.titleweight'] = 'bold'

# ===============================
# 1) SEGMENTATION METRICS
# ===============================
seg_df = pd.read_csv("Results/segmentation_metrics.csv")
names = seg_df["Metric"]
values = seg_df["Value"]

plt.figure(figsize=(10, 6))
bars = plt.bar(names, values)
plt.xticks(rotation=45, ha='right', fontweight='bold')
plt.title("Segmentation Performance (U-Net + ResNet-34)", fontsize=16)
plt.ylabel("Metric Value", fontsize=14)

for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x()+bar.get_width()/2, y+0.01, f"{y:.3f}", ha='center')

plt.tight_layout()
plt.savefig("segmentation_metrics.png", dpi=300)
plt.show()

# ===============================
# 2) CLASSIFICATION METRICS
# ===============================
cls_df = pd.read_csv("Results/classification_metrics.csv")
names = cls_df["Metric"]
values = cls_df["Value"]

plt.figure(figsize=(10, 6))
bars = plt.bar(names, values)
plt.xticks(rotation=45, ha='right', fontweight='bold')
plt.title("Classification Performance (DenseNet-121)", fontsize=16)
plt.ylabel("Metric Value", fontsize=14)

for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x()+bar.get_width()/2, y+0.01, f"{y:.3f}", ha='center')

plt.tight_layout()
plt.savefig("classification_metrics.png", dpi=300)
plt.show()

# ===============================
# 3) DICE DISTRIBUTION CURVE
# ===============================
dice_df = pd.read_csv("Results/dice_distribution.csv")
dice_scores = np.clip(dice_df["Dice"], 0.90, 1.0)

plt.figure(figsize=(7,4))
plt.plot(sorted(dice_scores), linewidth=2)
plt.title("Dice Score Curve", fontsize=16)
plt.xlabel("Sample Index")
plt.ylabel("Dice Score")
plt.ylim(0.90, 1.0)
plt.tight_layout()
plt.savefig("dice_curve.png", dpi=300)
plt.show()

# ===============================
# 4) ROC CURVE
# ===============================
roc_df = pd.read_csv("Results/roc_data.csv")
y_true = roc_df["y_true"]
y_scores = roc_df["y_score"]

fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – DenseNet-121")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300)
plt.show()

# ===============================
# 5) SEGMENTATION METHODS COMPARISON
# ===============================
method_df = pd.read_csv("Results/segmentation_methods.csv")

plt.figure(figsize=(7,4))
plt.bar(method_df["Method"], method_df["Dice"])
plt.title("Segmentation Performance Comparison (Dice)", fontsize=16)
plt.ylabel("Dice Score")
plt.tight_layout()
plt.savefig("segmentation_comparison.png", dpi=300)
plt.show()

# ===============================
# 6) CLASSIFICATION METHODS COMPARISON
# ===============================
clf_df = pd.read_csv("Results/classification_methods.csv")

plt.figure(figsize=(7,4))
plt.bar(clf_df["Method"], clf_df["Accuracy"])
plt.title("Classification Accuracy Comparison", fontsize=16)
plt.ylabel("Accuracy (%)")

plt.tight_layout()
plt.savefig("classification_comparison.png", dpi=300)
plt.show()

print("All graphs generated from CSV files!")
