# -*- coding: utf-8 -*-
"""
Tomato Leaf Disease Detection - YOLOv8 Training Pipeline
Author: DivaCodez
"""

import os
import glob
import yaml
from IPython.display import Image, display

# ==========================================
# 1. Environment & Dataset Setup
# ==========================================
# Check GPU availability
!nvidia-smi

# Extract dataset
!unzip -q /content/dataset.zip -d /content/custom_data

# Download dataset splitting helper
!wget -O /content/train_val_split.py https://raw.githubusercontent.com/EdjeElectronics/Train-and-Deploy-YOLO-Models/refs/heads/main/utils/train_val_split.py

# Perform train/validation split (90% train, 10% validation)
!python train_val_split.py --datapath="/content/custom_data/data" --train_pct=0.9

# Install required dependencies
!pip install -q ultralytics


# ==========================================
# 2. Dynamic YAML Configuration Generator
# ==========================================
def create_data_yaml(path_to_classes_txt, path_to_data_yaml):
    """
    Reads class labels from a text file and generates a YOLOv8 data.yaml config.
    """
    if not os.path.exists(path_to_classes_txt):
        print(f"Error: {path_to_classes_txt} not found!")
        return

    with open(path_to_classes_txt, 'r') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]

    data_config = {
        'path': '/content/data',
        'train': 'train/images',
        'val': 'validation/images',
        'nc': len(classes),
        'names': classes
    }

    with open(path_to_data_yaml, 'w') as f:
        yaml.dump(data_config, f, sort_keys=False)

    print(f"Created YAML config at {path_to_data_yaml}")


# Generate YAML configuration file
CLASSES_TXT_PATH = '/content/custom_data/data/classes.txt'
YAML_OUTPUT_PATH = '/content/data.yaml'

create_data_yaml(CLASSES_TXT_PATH, YAML_OUTPUT_PATH)

# Verify generated YAML
print('\n--- Generated data.yaml ---')
!cat /content/data.yaml


# ==========================================
# 3. Model Training (Iteration 3 - Optimized)
# ==========================================
# Training with 150 Epochs and Hyperparameter Data Augmentations
# to combat background confusion and improve disease spot detection
!yolo detect train \
    data=/content/data.yaml \
    model=yolov8n.pt \
    epochs=150 \
    imgsz=640 \
    degrees=15 \
    flipud=0.5 \
    fliplr=0.5 \
    hsv_v=0.4 \
    hsv_s=0.4 \
    conf=0.15 \
    mosaic=1.0


# ==========================================
# 4. Evaluation & Prediction
# ==========================================
# Run prediction on validation set using best trained weights
!yolo detect predict \
    model=runs/detect/train/weights/best.pt \
    source=data/validation/images \
    conf=0.15 \
    save=True

# Display sample predictions
print("\n--- Visualizing Sample Predictions ---")
for image_path in glob.glob('/content/runs/detect/predict/*.jpg')[:5]:
    display(Image(filename=image_path, height=400))


# ==========================================
# 5. Export Model Artifacts
# ==========================================
# Organize model weights and training logs for download
!mkdir -p /content/my_model
!cp /content/runs/detect/train/weights/best.pt /content/my_model/my_model.pt
!cp -r /content/runs/detect/train /content/my_model/train_logs

# Compress into zip archive
%cd /content/my_model
!zip -r /content/tomato_yolov8_model.zip my_model.pt train_logs
%cd /content

print("\nPipeline complete! Saved model archive to /content/tomato_yolov8_model.zip")
