# Bacterial-Flagellar-Motors-detection
# 🧠 Bacterial Flagellar Motor Detection in Cryo-Tomograms

This project aims to detect **bacterial flagellar motors** from cryo-electron tomography (Cryo-ET) data using deep learning. The workflow combines semantic segmentation and object detection to infer motor positions from 3D tomograms.

---

## 📌 Overview

Cryo-tomograms are 3D volumes represented as a stack of 2D slices. Flagellar motors are small, low-contrast structures that are hard to detect manually. We automate their localization with the following pipeline:

### Workflow:
1. **Segmentation (U-Net)** – Segment the bacterial body slice-by-slice.
2. **Membrane Detection (YOLO)** – Detect membrane regions in each 2D slice.
3. **Motor Localization** – Combine both outputs to estimate motor candidates.
4. **3D Consistency Check** – Track detected motor positions across slices and average them to remove duplicates.

---

## 🧪 Model Architecture

- **U-Net**: For 2D segmentation of bacterial cells in each slice.
- **YOLO**: For 2D detection of membrane boundaries in each slice.
- **Postprocessing**:
  - Track predicted motor locations across consecutive slices.
  - Compute the **mean position** for consistent 3D localization.

---

## ⚠️ Challenges

- **Small Object Detection**: Flagellar motors are extremely small and can resemble noise.
- **Low Signal-to-Noise Ratio**: Cryo-ET data is inherently noisy.
- **3D Consistency**: Motors should be localized accurately in 3D, not just per slice.
- **Limited Annotations**: Manual labeling is time-consuming and scarce.

---

## 📂 Folder Structure

project/
│
├── data/ # Preprocessed tomogram slices
├── models/ # Trained model weights
├── scripts/ # Training & inference scripts
│ ├── train_unet.py
│ ├── train_yolo.py
│ ├── infer_pipeline.py
│
├── utils/ # Helper functions for tracking, averaging, visualization
├── results/ # Output segmentations and motor detections
└── README.md 

