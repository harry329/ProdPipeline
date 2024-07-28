## Model Name: 
Census Income Prediction Model

## Overview:
This model is designed to predict whether an individual's income exceeds $50,000 per year based on census data. The model is trained using various demographic and employment-related features.

## Model Details:
Algorithm: The specific algorithm used for training is determined by the train_model function from the starter.ml.model module.
Training Data: The model is trained on census data split into training (80%) and testing (20%) sets.
Features: The model uses both categorical and numerical features:
Categorical Features: workclass, education, marital-status, occupation, relationship, race, sex, native-country
Numerical Features: Features not explicitly listed as categorical are considered numerical.
Label: The target variable is salary, which indicates whether the individual's income exceeds $50,000 per year.

## Preprocessing:
Categorical features are processed using encoding techniques.
Data is split into training and testing sets.
The process_data function is used to handle preprocessing for both training and testing datasets.
Usage:

## Training: 
The model can be trained using the train_save_model function, which saves the trained model to a specified path.
Inference: Predictions can be made using the make_prediction function, which loads the trained model and predicts based on input features.

## Performance Metrics:
The model's performance metrics (e.g., accuracy, precision, recall) are evaluated using the test set. These metrics should be documented based on the evaluation performed after training.
