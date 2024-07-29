import os
import joblib
import unittest

import numpy as np
import pandas as pd

from starter.starter.ml.data import process_data
from starter.starter.train_model import train_save_model, make_prediction, cat_features


class TestModelTraining(unittest.TestCase):
    def test_train_save_model(self):
        # Ensure the model training and saving function runs without errors
        train_save_model()
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '..', 'starter', 'model',
                                  'trained_model.joblib')
        self.assertTrue(os.path.exists(model_path))
        # Load the model to check if it's saved correctly
        model = joblib.load(model_path)
        self.assertIsNotNone(model)

    def test_make_prediction(self):
        # Generate a prediction and check its type
        prediction = make_prediction()
        self.assertIsInstance(prediction, np.ndarray)
        self.assertEqual(prediction.shape, (1,))

    def test_process_data(self):
        sample_data = {
            "age": [25, 50],
            "workclass": ["Private", "Self-emp-not-inc"],
            "fnlwgt": [226802, 89814],
            "education": ["11th", "HS-grad"],
            "education-num": [7, 9],
            "marital-status": ["Never-married", "Married-civ-spouse"],
            "occupation": ["Machine-op-inspct", "Farming-fishing"],
            "relationship": ["Own-child", "Husband"],
            "race": ["Black", "White"],
            "sex": ["Male", "Male"],
            "capital-gain": [0, 0],
            "capital-loss": [0, 0],
            "hours-per-week": [40, 50],
            "native-country": ["United-States", "United-States"],
            "salary": ["<=50K", ">50K"]
        }
        df = pd.DataFrame(sample_data)
        X, y, encoder, lb = process_data(df, categorical_features=cat_features, label="salary", training=True)
        self.assertEqual(X.shape[0], 2)
        self.assertEqual(y.shape[0], 2)
        self.assertIsNotNone(encoder)
        self.assertIsNotNone(lb)
