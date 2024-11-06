# CIFAR-10 Image Classifier

This project implements an image classification model trained on the CIFAR-10 dataset using a Convolutional Neural Network (CNN) built with PyTorch. The model is trained to recognize ten different classes of objects, including planes, cars, birds, and more.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Training](#training)
- [Testing](#testing)

## Project Overview

The main components of this project include:
- **Data loading and preprocessing**: Loads the CIFAR-10 dataset, applies transformations, and creates data loaders for training and testing.
- **Model training**: Trains a CNN on the CIFAR-10 training data.
- **Model evaluation**: Tests the model on the test data and computes overall and class-specific accuracy metrics.
- **Model saving and loading**: Allows saving the trained model's parameters and loading them for later use.

## Dataset

This project uses the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), a collection of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is split into 50,000 training images and 10,000 test images.

Classes:
- Plane
- Car
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/ManaPoustizadeh/image-classifier.git
    cd image-classifier
    ```

2. Install the required dependencies

## Usage

Run the main script to train and evaluate the model:
```bash
python -m classifier.main  
```

## Project Structure

There is a sample CNN for introduction purposes inside the `sample_cnn` folder.
The classifier folder includes CNN definition and functions needed to load the data, train the network, and test it. Below is a more detailed explanation.

### Configuration

The `ClassificationConfig` class in `main.py` allows setting the following parameters:

batch_size: Number of samples per gradient update (default: 4).
epoch_size: Number of epochs to train the model (default: 2).
transform: Transformations applied to the images, including tensor conversion and normalization.
You can modify these parameters in ClassificationConfig to customize the training process.

### Training

The train_network method in Classifier class trains the CNN model on the CIFAR-10 training data. It includes:

* Loss function: Cross-Entropy Loss for multi-class classification.
* Optimizer: SGD (Stochastic Gradient Descent) with a learning rate of 0.001 and momentum of 0.9.
Training prints the loss at every 2000 mini-batches for monitoring.

### Testing

The test_network method evaluates the trained model on the CIFAR-10 test dataset, reporting accuracy based on correctly classified images.
