Of course! Here is a comprehensive README.md file generated from the provided notebook pages.

---

# COVID-19 Image Classification using Deep Learning

This project, developed as part of the Master 2 in Cybersecurity and Artificial Intelligence program at the University of Ain Temouchent, focuses on the classification of chest X-ray images to detect COVID-19 using various deep learning techniques.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Class Distribution](#class-distribution)
  - [Sample Images](#sample-images)
- [Methodology](#methodology)
  - [Image Segmentation](#image-segmentation)
  - [Data Augmentation](#data-augmentation)
  - [Modeling Approaches](#modeling-approaches)
- [Model Performance](#model-performance)
  - [1. Simple CNN](#1-simple-cnn)
  - [2. CNN Feature Extractor + XGBoost Classifier](#2-cnn-feature-extractor--xgboost-classifier)
  - [3. Transfer Learning: MobileNet](#3-transfer-learning-mobilenet)
  - [4. Transfer Learning: Xception](#4-transfer-learning-xception)
- [Results Summary](#results-summary)
- [Model Interpretation with Grad-CAM](#model-interpretation-with-grad-cam)
- [Interactive Demo with Gradio](#interactive-demo-with-gradio)
- [Setup and Usage](#setup-and-usage)
- [Authors](#authors)

## Project Overview
The primary goal of this project is to build and evaluate several deep learning models for the binary classification of chest X-rays into 'COVID-19' and 'Normal' categories. The project explores custom CNN architectures, hybrid models combining CNNs with traditional machine learning, and transfer learning using state-of-the-art pre-trained models.

## Dataset
The project utilizes the **COVID19 dataset** from Kaggle.


The dataset contains chest X-ray images categorized into two classes: `covid19` and `normal`.

### Class Distribution
The dataset is well-balanced, with a nearly equal number of images in each class, which is ideal for training a classification model without significant class bias.



### Sample Images
Below are sample images from each class.

| COVID-19 Sample 1 | COVID-19 Sample 2 |
| :---: | :---: |
|  |  |

| Normal Sample 1 | Normal Sample 2 |
| :---: | :---: |
|  |  |


## Methodology

### Image Segmentation
Initial experiments were conducted to explore image segmentation as a preprocessing step. Techniques like **Fuzzy C-Means clustering** and **DBSCAN** were applied to identify and isolate regions of interest (e.g., lungs) within the X-ray images.

### Data Augmentation
To improve model generalization and prevent overfitting, data augmentation techniques were applied. These include random zooming, shearing, and horizontal flipping to create a more diverse set of training images.

### Modeling Approaches
Four distinct modeling strategies were implemented and evaluated:
1.  **Simple CNN:** A custom Convolutional Neural Network built from scratch.
2.  **CNN + XGBoost:** A hybrid approach using the custom CNN as a feature extractor and an XGBoost model as the final classifier.
3.  **MobileNet (Transfer Learning):** A pre-trained MobileNet model fine-tuned for the classification task.
4.  **Xception (Transfer Learning):** A pre-trained Xception model fine-tuned for the classification task.

## Model Performance

### 1. Simple CNN
A simple CNN with three convolutional layers was trained for 15 epochs.

- **Final Accuracy:** **89.8%**

- **Confusion Matrix:**
  | | Predicted: covid19 | Predicted: normal |
  | :--- | :---: | :---: |
  | **Actual: covid19** | 21 | 4 |
  | **Actual: normal**| 1 | 23 |

### 2. CNN Feature Extractor + XGBoost Classifier
Features were extracted from the final convolutional layer of the Simple CNN and used to train an XGBoost classifier.

- **Validation Accuracy:** **93.33%**

- **Confusion Matrix:**
  | | Predicted: Class 0 | Predicted: Class 1 |
  | :--- | :---: | :---: |
  | **Actual: Class 0** | 28 | 2 |
  | **Actual: Class 1** | 2 | 28 |


### 3. Transfer Learning: MobileNet
The pre-trained MobileNet model was used with its base layers frozen. The model was trained for 30 epochs.

- **Final Accuracy:** **53.3%**
- **Note:** This model performed poorly on the validation set, achieving an accuracy only slightly better than random chance, despite high training accuracy. This indicates significant overfitting or other training issues.

- **Confusion Matrix:**
  | | Predicted: covid19 | Predicted: normal |
  | :--- | :---: | :---: |
  | **Actual: covid19** | 16 | 14 |
  | **Actual: normal**| 14 | 16 |

### 4. Transfer Learning: Xception
The pre-trained Xception model was fine-tuned for the task, also for 30 epochs. This model demonstrated superior performance.

- **Validation Accuracy:** **100%**

- **Confusion Matrix:**
  | | Predicted: covid19 | Predicted: normal |
  | :--- | :---: | :---: |
  | **Actual: covid19** | 30 | 0 |
  | **Actual: normal**| 0 | 30 |

## Results Summary

| Model | Validation Accuracy | Key Observation |
| :--- | :---: | :--- |
| Simple CNN | 89.8% | Good baseline performance. |
| CNN + XGBoost | 93.3% | Hybrid model improved upon the baseline. |
| MobileNet | 53.3% | Performed poorly, likely due to overfitting. |
| **Xception** | **100%** | **Achieved perfect accuracy on the validation set.** |

The **Xception** model provided the best results, highlighting the power of transfer learning with deep, sophisticated architectures for medical imaging tasks.

## Model Interpretation with Grad-CAM
To understand the decision-making process of the model, Gradient-weighted Class Activation Mapping (Grad-CAM) was applied. This technique generates a heatmap that highlights the regions of the input image most influential for a given prediction. The heatmap correctly identified areas within the lungs as critical for classifying an image as COVID-19.



## Interactive Demo with Gradio
A simple web interface was built using **Gradio** to allow for interactive, real-time prediction on new X-ray images using the trained models.


## Authors
This project was created by:
- **Kaddache mohammed el amine**
- **Elmeguenni nabil**
