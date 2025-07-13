# Deep-Learning-Project
# 🧠 OCT Image Classification Model 

This repository contains a complete **deep learning image classification** pipeline using **TensorFlow/Keras**. The goal: classify Optical Coherence Tomography (OCT) images into four categories—**CNV**, **DME**, **DRUSEN**, and **NORMAL**.

---

## 🎯 Project Overview

- **Dataset**: OCT scans (CNV, DME, DRUSEN, NORMAL) sourced from [Mendeley OCT2017 Dataset V2].
- **Task**: Build a convolutional neural network (CNN) using **transfer learning**.
- **Deliverables**:
  - A trained classification model
  - Visualization of training history (accuracy & loss plots)
  - Sample predictions
  - Saved model ready for further tasks or deployment

---

## 🛠️ Repository Structure
task_2_oct_classification/
├── data/
│ ├── train/ # Training images (subfolders CNV, DME, DRUSEN, NORMAL)
│ └── test/ # Test images
│
├── train_model.ipynb # Full notebook with model building & evaluation
├── model.py # Python script version 
└── README.md # This documentation

## Run the notebook

1. Load and preprocess images

2. Build a CNN using VGG16 (or another pretrained backbone)

3. Train and validate the model

4. Evaluate on test set and visualize performance

5. Save the final model to oct_cnn_model.h5

## DATASET : https://data.mendeley.com/datasets/rscbjbr9sj/2
## IMAGES 


Built by Nisha during the CODTECH internship
Feel free to fork, replicate, and enhance this work 😊
