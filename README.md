# Playground ML

## Overview
**Playground ML** is a multi-functional machine learning application built with Streamlit. It offers users a variety of tools to explore and experiment with different machine learning models and tasks, including regression, classification, and nails detection using images and videos. The interface provides a simple way to interact with these models, test predictions, and visually explore the results. The playground is accessible here: https://projectml-kuntbmofsprgvuy7icvpfc.streamlit.app/ "

## Features

- **Home Page:** A visual introduction to the application.
- **Regression Module:** Allows users to perform regression analysis.
- **Classification Module:** Facilitates data preparation, analysis, deep learning experiments, and usage of pre-trained models.
- **Nails Detection:** Detect nails using either static images or live video streaming.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Modules](#modules)
   - [Home](#home)
   - [Regression](#regression)
   - [Classification](#classification)
   - [Nails Detection](#nails-detection)
4. [Requirements](#requirements)
5. [Contributing](#contributing)
6. [License](#license)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/playground-ml.git
   
2. Navigate to the project directory:
    ```bash
   cd playground-ml
   
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
4. Run the application:
    ```bash
    streamlit run main.py

## Modules

### Home

- The **Home Page** serves as a visual introduction to the app, showcasing images related to the project.

### Regression

- The **Regression Module** allows users to perform regression analysis using their own dataset. The module guides users through:
  - Loading data.
  - Selecting features.
  - Running regression models.
  
### Classification

- The **Classification Module** consists of several tabs to assist users in preparing and analyzing data for classification tasks. The tabs are as follows:
  
  1. **Preparation**:
     - Data preprocessing steps like handling missing values, data cleaning, and preparing the dataset for analysis.

  2. **Analysis**:
     - Perform exploratory data analysis (EDA) to gain insights into your dataset.

  3. **Play with parameters**:
     - Allows users to tune model parameters and observe their impact on model performance.

  4. **Try a DNN**:
     - Experiment with Deep Neural Networks (DNNs) for classification tasks. Users can try different architectures and hyperparameters.

  5. **Use your models**:
     - Load and test models that have been previously trained, and apply them to new datasets for predictions.

### Nails Detection

- The **Nails Detection Module** provides tools to detect nails in both images and videos. It contains two tabs:

  1. **Image**:
     - Upload an image, and the model will detect nails within the image and highlight the results.

  2. **Video**:
     - Detect nails in real-time using a live video stream. The app processes the video input, applies the nails detection model, and displays the results directly on the stream. A demo GIF related to nails detection is also shown.

## Requirements

Python 3.12
Streamlit
OpenCV
PIL
Additional dependencies listed in requirements.txt
