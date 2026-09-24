import unittest
import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data_preprocessing import ChurnDataPreprocessor
from predict import ChurnPredictor

class TestMLPipeline(unittest.TestCase):

    def test_preprocessor_artifact_exists(self):
        preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.joblib')
        self.assertTrue(os.path.exists(preprocessor_path), "Preprocessor joblib artifact missing.")

    def test_model_artifact_exists(self):
        model_path = os.path.join(MODELS_DIR, 'churn_model.joblib')
        self.assertTrue(os.path.exists(model_path), "Model joblib artifact missing.")

    def test_churn_predictor_single(self):
        predictor = ChurnPredictor(models_dir=MODELS_DIR)
        sample = {
            'CreditScore': 600,
            'Geography': 'France',
            'Gender': 'Male',
            'Age': 40,
            'Tenure': 5,
            'Balance': 50000.0,
            'NumOfProducts': 2,
            'HasCrCard': 1,
            'IsActiveMember': 1,
            'EstimatedSalary': 60000.0
        }
        res = predictor.predict_single(sample)
        self.assertIn('churn_prediction', res)
        self.assertIn(res['churn_prediction'], [0, 1])
        self.assertGreaterEqual(res['churn_probability'], 0.0)
        self.assertLessEqual(res['churn_probability'], 1.0)
        self.assertIn(res['risk_level'], ['Low', 'Medium', 'High'])

if __name__ == '__main__':
    unittest.main()
