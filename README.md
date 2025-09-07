# Satellite-Image-Based-Disaster-Detection
Satellite Image-Based Flood Detection and Mapping using Sentinel-1 SAR data and U-Net deep learning model. Provides rapid disaster mapping for flood-affected regions.

---

This project uses **Sentinel-1 SAR satellite imagery** and **deep learning (U-Net, YOLOv8)** to detect and map flood-affected regions.  
It enables **rapid disaster response** by generating flood extent maps for emergency management and relief planning.  

---

## Objectives
- Detect flood-affected areas from Sentinel-1 SAR images.
- Generate pixel-level flood maps using U-Net segmentation.
- Compare real-time detection performance with YOLOv8.
- Provide tools for disaster management and rapid damage mapping.

---

## Dataset
- **SEN12-FLOOD** (Sentinel-1 SAR + flood/non-flood labels)  

---

## Models Used
- **U-Net** → Pixel-level segmentation for flood extent mapping.
- **YOLOv8** → Object detection for bounding-box flood area estimation.

---

## Workflow
1. Data Collection (SEN12-FLOOD dataset)  
2. Preprocessing (cloud masking, normalization, patches)  
3. Model Training (U-Net / YOLOv8)  
4. Evaluation (IoU, Dice Score, Precision-Recall)  
5. Visualization (Flood extent overlay on satellite imagery)  

---

## Tech Stack
- Python (PyTorch, TensorFlow)
- OpenCV, Rasterio, GDAL
- Jupyter Notebooks / Google Colab

---

## Results (to be added after experiments)
- Flood extent maps  
- Quantitative metrics  

---
