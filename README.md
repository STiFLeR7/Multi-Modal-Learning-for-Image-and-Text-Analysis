# Multi-Modal Learning for Image and Text Analysis

## Overview

This repository demonstrates a **Multi-Modal Learning** approach for **Image and Text Analysis** using **PyTorch**. The solution integrates **Computer Vision (CV)** and **Natural Language Processing (NLP)** to analyze the **COCO dataset**, combining visual and textual features to make accurate predictions. The model is based on **ResNet18**, fine-tuned for image classification tasks.

---

## Table of Contents

1. [Install Dependencies](#install-dependencies)
2. [Usage](#usage)
3. [Directory Structure](#directory-structure)
4. [Dependencies](#dependencies)
5. [Training the Model](#training-the-model)
6. [Inference and Predictions](#inference-and-predictions)
7. [Results and Evaluation](#results-and-evaluation)
8. [License](#license)

---

## Install Dependencies

To get started with the project, first clone the repository to your local machine:

```bash
git clone https://github.com/STiFLeR7/Multi-Modal-Learning-for-Image-and-Text-Analysis.git
cd Multi-Modal-Learning-for-Image-and-Text-Analysis

pip install -r requirements.txt

python model.py

python inference_save.py

python visualize.py

Multi-Modal-Learning-for-Image-and-Text-Analysis/
│
├── annotations_fixed.json            # Corrected annotations for the dataset
├── coco_resnet18.pth                # Trained model weights
├── inference_save.py                # Script for inference and saving predictions
├── model.py                         # Model training script
├── predictions.json                 # Predictions after running inference
├── preprocessed_data/               # Folder with preprocessed image batches
│   ├── images_batch_0.npy
│   ├── images_batch_1.npy
│   └── annotations.json
├── requirements.txt                 # Dependencies file
├── src/                             # Main code directory
│   ├── coco_loader.py               # Dataset loader for COCO data
│   ├── txt-json.py                  # Script for combining annotations from batches
│   └── visualize.py                 # Visualization script for results
└── README.md                        # Project documentation

pip install -r requirements.txt

python model.py

python inference_save.py

[
  {
    "image_id": 1,
    "predicted_class": 4,
    "confidence": 0.95
  },
  {
    "image_id": 2,
    "predicted_class": 12,
    "confidence": 0.87
  }
]


---

### Steps to Proceed:
1. Copy and paste this content into a new file called `README.md`.
2. Commit and push it to your GitHub repository.

This `README.md` contains all necessary sections, including project setup, usage instructions, dependencies, and directory structure. Let me know if you need any further modifications!
