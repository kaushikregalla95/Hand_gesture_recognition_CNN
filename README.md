# Hand Gesture Recognition with CNNs

**Student:** Kaushik Reddy Ragalla  
**Project:** #54 — Biometric and Human-Centered Recognition  
**Course:** Machine Learning — Phase 2 Project  

---

## Project Overview

This project implements a complete CNN-based pipeline to classify **10 hand gestures** captured with a Leap Motion infrared camera. Three models are trained and compared:

| Model | Test Accuracy | Type |
|-------|-------------|------|
| Custom CNN | 99.90% | Trained from scratch |
| MobileNetV2 | 99.93% | Transfer learning + fine-tuning |
| VGG16 | 99.97% | Transfer learning |

---

## Dataset

**LeapGestRecog** — GTI-UPM, Universidad Politécnica de Madrid  
🔗 https://www.kaggle.com/datasets/kaushikfreddyregalla/hand-gesture-recognition-algorithm-using-cnn

| Attribute | Value |
|-----------|-------|
| Total images | 20,000 |
| Classes | 10 gesture types |
| Subjects | 10 subjects (00–09) |
| Image type | Grayscale / Near-infrared PNG |
| Class distribution | Perfectly balanced (2,000 per class) |
| License | CC0 Public Domain |

**Gesture Classes:**
`01_palm`, `02_l`, `03_fist`, `04_fist_moved`, `05_thumb`, `06_index`, `07_ok`, `08_palm_moved`, `09_c`, `10_down`

---

## Repository Structure

```
Hand_gesture_recognition/
├── README.md
├── notebook/
│   ├── 01-setup-exploratory-data-analysis.ipynb
│   ├── 02-preprocessing-data-augmentation.ipynb
│   ├── 03-custom-cnn-model.ipynb
│   ├── 04-transfer-learning-mobilenetv2-vgg16.ipynb
│   ├── 05-model-evaluation.ipynb
│   └── 06-grad-cam-error-analysis.ipynb
└── figures/
    ├── sample_images.png
    ├── class_distribution.png
    ├── training_curves_cnn.png
    ├── training_curves_mobilenet.png
    ├── training_curves_vgg16.png
    ├── confusion_matrix_cnn.png
    ├── confusion_matrix_mobilenet.png
    ├── confusion_matrix_vgg16.png
    ├── roc_curves.png
    ├── gradcam_examples.png
    └── misclassified_examples.png
```

---

## Notebook Descriptions

| Notebook | Description |
|----------|-------------|
| 01 — Setup & EDA | Library imports, dataset loading, class distribution chart, sample image grid |
| 02 — Preprocessing | Image loading, resizing (128×128), normalization, 70/15/15 split, data augmentation |
| 03 — Custom CNN | 4-block CNN architecture, BatchNorm, Dropout, training curves, test accuracy |
| 04 — Transfer Learning | MobileNetV2 (head training + fine-tuning top 30 layers) and VGG16 comparison |
| 05 — Model Evaluation | Confusion matrices, classification reports, ROC curves, model comparison table |
| 06 — Grad-CAM | SmoothGrad saliency maps, error analysis, misclassified examples |

---

## How to Run on Kaggle

### Step 1 — Add Dataset
In each notebook → **+ Add Input** → search `kaushikfreddyregalla hand gesture recognition` → Add

### Step 2 — Enable GPU
**Session Options → Accelerator → GPU T4 x2**

### Step 3 — Run in Order
Run notebooks **01 → 02 → 03 → 04 → 05 → 06** in sequence.

For notebooks 03–06, attach the output of notebook 02 as input:
- **+ Add Input → Your Work → 02 — Preprocessing & Data Augmentation**

For notebooks 05–06, also attach notebook 04 output:
- **+ Add Input → Your Work → 04 — Transfer Learning**

---

## Methodology

1. **Data loading** — Scan subject/class folder hierarchy, collect image paths and labels
2. **Preprocessing** — Grayscale imread → resize 128×128 → normalize [0,1]
3. **Split** — Stratified 70% train / 15% validation / 15% test
4. **Augmentation** — Rotation ±15°, shift ±10%, zoom ±10%, horizontal flip (training only)
5. **Custom CNN** — 4 conv blocks (32→64→128→256 filters), BatchNorm, Dropout, GAP, Dense
6. **Transfer learning** — MobileNetV2 & VGG16; Phase 1: frozen base; Phase 2: fine-tune top 30 layers
7. **Evaluation** — Accuracy, confusion matrix, classification report, ROC/AUC
8. **Interpretability** — SmoothGrad saliency maps on MobileNetV2

---

## Key Results

- **Best model:** VGG16 with **99.97% test accuracy**
- **ROC AUC:** 1.000 for all 10 classes (MobileNetV2)
- **Error rate:** Only 2 misclassifications out of 3,000 test images (MobileNetV2)
- **Most confused gestures:** `04_fist_moved` ↔ `09_c` (visually similar hand positions)

---

## Requirements

```
tensorflow>=2.10
opencv-python>=4.7
scikit-learn>=1.2
matplotlib>=3.6
seaborn>=0.12
pandas>=1.5
numpy>=1.23
```

---

## References

- Molina, J. et al. (2019). *LeapGestRecog dataset*. GTI-UPM, Kaggle.
- Sandler, M. et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR.
- Simonyan, K. & Zisserman, A. (2015). *Very Deep Convolutional Networks (VGG)*. ICLR.
- Smilkov, D. et al. (2017). *SmoothGrad: removing noise by adding noise*. ICML Workshop.
