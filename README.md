# ResNeXt Transfer Learning for Multi-Class Image Classification

This notebook demonstrates how to apply **transfer learning using ResNeXt** for a multi-class image classification task. The approach leverages pretrained weights from PyTorch's `torchvision.models` to train a classifier on a custom dataset with minimal compute.

## 🖼️ Dataset

- **Classes:** `dogs`, `food`, `vehicles`
- **Input Format:** Images organized in folders inside a ZIP archive
- **Preprocessing:** All images resized to 64x64

## 🔍 Workflow

- Extract images from zip
- Load dataset using PIL and convert to NumPy arrays
- Normalize images and encode labels
- Split into training and testing sets
- Use `ResNeXt101_32x8d` from `torchvision.models` with a custom classification head

## 🧠 Model Details

- **Backbone:** ResNeXt-101 32x8d (pretrained on ImageNet)
- **Modifications:** Final layer replaced to classify 3 custom categories
- **Loss Function:** Cross-Entropy
- **Optimizer:** Adam

## 📊 Evaluation Metrics

- Accuracy
- Precision, Recall, F1 Score (Macro)
- Confusion Matrix

## 📈 Visualization

- Training loss and accuracy curves
- Heatmap of confusion matrix
- Classification report

## 🧰 Requirements

```bash
pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn pillow tqdm
