#Hair Type Classification CNN — Module 8 (ML Zoomcamp)
This repository contains my Module 8 project from the Machine Learning Zoomcamp by DataTalks. The goal of this project is to build a Convolutional Neural Network (CNN) from scratch to classify hair types (straight vs curly) using PyTorch.

This project demonstrates:

Image data preprocessing and normalization

Building and training a CNN for binary classification

Using data augmentation to improve model generalization

Tracking training and validation metrics

Analysis of model performance


Project: Hair Type Classification CNN
Dataset

The dataset contains ~1000 images of hair, split into train and test folders.

Dataset source (for reference): Hair Type Dataset on Kaggle

Folder structure:

data/
  train/
    straight/
    curly/
  test/
    straight/
    curly/



Features

##CNN architecture:

Input: (3, 200, 200) images

1 Convolutional layer (32 filters, 3x3 kernel) + ReLU + MaxPool

Flatten → Fully Connected Layer (64 neurons) + ReLU

Output layer: 1 neuron + sigmoid (binary classification)

Optimizer: SGD with lr=0.002, momentum=0.8

Loss function: nn.BCEWithLogitsLoss()

Training: 20 epochs (10 initial + 10 with augmentation)

##Data augmentation:

Random rotation (±50°)

Random resized crop

Random horizontal flip

Results

Median training accuracy (first 10 epochs): 0.40

Standard deviation of training loss: 0.171

Mean test loss after augmentation: 0.08

Average test accuracy (last 5 epochs): 0.68

#3How to Run

Clone the repo:

git clone https://github.com/YourUsername/Module8_HairType_CNN.git


Upload the data.zip dataset and extract it into the data/ folder.

Open the Colab notebook: Module8_HairType_CNN.ipynb.

Run all cells sequentially.

Note: Training can be run on GPU (Colab runtime) for faster performance.

Learning Outcomes

Practical understanding of CNNs for image classification

Experience in PyTorch for building and training deep learning models

Hands-on experience with data augmentation and model evaluation metrics
