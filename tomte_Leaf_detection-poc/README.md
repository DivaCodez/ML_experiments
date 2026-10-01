# 🍅 Tomato Leaf Disease Detection (YOLOv8 & Computer Vision)

## 🎯 Overview & Objective

This module focuses on **Object Detection** in agricultural Computer Vision. The goal is to build an end-to-end pipeline to detect and classify tomato plant leaf diseases (*Early Blight* vs. *Late Blight*) using **YOLOv8**, custom bounding-box annotations via **Label Studio**, and an interactive web interface powered by **Gradio**.

---

## 🛠️ Stack & Methodology

* **Data Source:** Kaggle Tomato Leaf Disease Dataset
* **Annotation & Refinement:** Label Studio (Custom YOLO format bounding boxes)
* **Model Architecture:** YOLOv8 (Ultralytics)
* **Augmentations & Training:** PyTorch & Ultralytics Training Engine
* **Deployment Interface:** Gradio

---

## 🔬 Iterative Experimentation & Performance Evolution

To address challenge areas like background confusion and small spot detection, the model went through three primary training iterations:

| Iteration | Setup & Augmentations | Early Blight Recall | Late Blight Recall | Key Insights / Progress |
| --- | --- | --- | --- | --- |
| **Iteration 1** | Baseline (60 Epochs) | Low | Low | Struggled to detect small disease spots and distinct patterns. |
| **Iteration 2** | Extended Epochs & Confidence Adjustment | 9% | 12% | Severe background confusion (False Negatives > 85%).|
| **Iteration 3 (Final)** | 150 Epochs + Augmentations (`degrees=15`, `flipud=0.5`, `fliplr=0.5`, `hsv_v=0.4`, `hsv_s=0.4`, `mosaic=1.0`) | **82%**<br> | **72%**<br> | **Major Breakthrough:** Clear disease differentiation and strong spot sensitivity.

 |

> **Key Learning:** Applying Mosaic augmentation alongside HSV lighting adjustments significantly improved the model's ability to isolate disease spots from complex background noise.
> 
> 

---

## 📁 Folder Structure

```text
.
├── data.zip                  # Kaggle images & Label Studio YOLO exports
├── weights/                  # Trained YOLOv8 model weights (.pt)
├── src/                  # Kaggle images & Label Studio YOLO exports
│   ├── main.py/
│   ├── train_val_split/
    └── train_yolo/       # Model training script with augmentations    
├── requirements.txt           
└── README.md            
```

---

## 🚀 Quick Execution

### 1. Model Training

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='dataset/data.yaml',
    epochs=150,
    imgsz=640,
    degrees=15,
    flipud=0.5,
    fliplr=0.5,
    hsv_v=0.4,
    hsv_s=0.4,
    conf=0.15,
    mosaic=1.0
)

```

### 2. Launch Gradio App

```python
python app.py

```

---

### 1. Prerequisites

Ensure you have Python 3.8+ installed along with PyTorch:

```bash
pip install ultralytics gradio label-studio-sdk opencv-python

```

### 2. Dataset Setup

1. Download the base dataset from Kaggle.
2. Annotate custom bounding boxes around disease spots using **Label Studio**.
3. Export annotations in **YOLO format** and organize your `data.yaml` file:

```yaml
train: ../dataset/train/images
val: ../dataset/valid/images

nc: 2
names: ['Early blight', 'Late blight']

```

---

## 🏋️ Training the Model

To run the optimized training pipeline (Iteration 3):

```python
from ultralytics import YOLO

# Load pre-trained model
model = YOLO('yolov8n.pt')

# Train with hyperparameter augmentations
model.train(
    data='data.yaml',
    epochs=150,
    imgsz=640,
    degrees=15,
    flipud=0.5,
    fliplr=0.5,
    hsv_v=0.4,
    hsv_s=0.4,
    conf=0.15,
    mosaic=1.0
)

```

---

## 🖥️ Running the Gradio Web App

Launch the interactive Web UI to test predictions on new images:

```python
python app.py 

```
