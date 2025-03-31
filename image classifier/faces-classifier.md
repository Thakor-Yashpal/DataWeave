# Animal Faces Image Classifier

This project implements a convolutional neural network (CNN) for classifying images of animal faces from the AFHQ dataset (cats, dogs, and wild animals). It utilizes PyTorch for model building, training, and evaluation, and leverages the Kaggle API to download the dataset.

## Overview

*   **Dataset:**  AFHQ (Animal Faces HQ) dataset, containing images of cats, dogs, and wild animals.  Downloaded directly from Kaggle using the Kaggle API.
*   **Model:** A simple CNN architecture with convolutional layers, max-pooling, ReLU activations, and fully connected layers for classification.
*   **Framework:** PyTorch
*   **Libraries:**  `torch`, `torchvision`, `pandas`, `numpy`, `matplotlib`, `PIL`, `opendatasets`

## Setup

1.  **Install Dependencies:**

    ```bash
    pip install opendatasets torch torchvision scikit-learn matplotlib
    ```

2.  **Kaggle Credentials:** You will need a Kaggle API key to download the dataset.

    *   Create a Kaggle account (if you don't have one).
    *   Go to your account settings on Kaggle and generate a new API token. This will download a `kaggle.json` file.
    *   You'll be prompted for your Kaggle username and key when running the notebook. This information is used by the `opendatasets` library to authenticate and download the dataset.

## Usage

1.  **Run the Notebook:**

    *   Upload the `image_classifier_.ipynb` notebook to Google Colab.
    *   Execute the notebook cells in order. It will guide you through downloading the data, training the model, and evaluating its performance.

## Google Colab

You can run this notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SZu_u8eMv4tUpySydgjixL3HLH4CJqD2)

## Notebook Structure

*   **Data Download and Preparation:**  Downloads the AFHQ dataset from Kaggle, creates Pandas DataFrames for train/test/validation splits.
*   **Data Loading and Transformation:** Defines a custom PyTorch Dataset for loading and transforming images.
*   **Model Definition:**  Defines the CNN architecture using PyTorch's `nn.Module`.
*   **Training Loop:**  Trains the model using Adam optimizer and CrossEntropyLoss. Includes validation to track performance.
*   **Evaluation:** Evaluates the trained model on the test dataset.
*   **Visualization:** Plots training and validation loss and accuracy.
*   **Prediction Example:**  Includes a function for predicting the class of a single image.

## Improvements

*   **Data Augmentation:** Implement data augmentation techniques to improve model generalization.
*   **Model Architecture:** Experiment with different CNN architectures, such as ResNet or EfficientNet.
*   **Hyperparameter Tuning:**  Optimize hyperparameters like learning rate, batch size, and number of epochs.
*   **Regularization:** Add regularization techniques (dropout, weight decay) to prevent overfitting.
*   **More Extensive Evaluation:** Conduct a more thorough evaluation of the model, including metrics like precision, recall, and F1-score.
