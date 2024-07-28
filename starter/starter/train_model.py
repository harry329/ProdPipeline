# Script to train machine learning model.
import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from starter.starter.ml.data import process_data
from starter.starter.ml.model import train_model, inference

# Add the necessary imports for the starter code.

# Add code to load in the data.

current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the census.csv file
csv_path = os.path.join(current_dir, '..', 'data', 'census.csv')
data = pd.read_csv(csv_path)
print(csv_path)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.

X_test, y_test, encoder_test, lb_test = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)


def train_save_model():
    model = train_model(X_train, y_train)
    model_path = os.path.join(current_dir, '..', 'model', 'trained_model.joblib')
    joblib.dump(model, model_path)


def make_prediction(X=None):
    model_path = os.path.join(current_dir, '..', 'model', 'trained_model.joblib')
    model = joblib.load(model_path)
    if X is None:
        X = X_test[0].reshape(1, -1)
    prediction = inference(model, X)
    print(prediction)
    return prediction
