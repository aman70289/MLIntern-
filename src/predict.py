import os
import joblib
import pandas as pd
import numpy as np

class ChurnPredictor:
    """
    Inference class for loading serialized joblib artifacts and running predictions.
    """
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.preprocessor = None
        self.model = None
        self.load_artifacts()

    def load_artifacts(self):
        preprocessor_path = os.path.join(self.models_dir, 'preprocessor.joblib')
        model_path = os.path.join(self.models_dir, 'churn_model.joblib')

        if not os.path.exists(preprocessor_path) or not os.path.exists(model_path):
            raise FileNotFoundError("Model or Preprocessor artifact missing. Please run train.py first.")

        self.preprocessor = joblib.load(preprocessor_path)
        self.model = joblib.load(model_path)

    def predict_single(self, input_dict: dict) -> dict:
        """
        Accepts a dictionary representing a single customer record and returns prediction details.
        """
        df = pd.DataFrame([input_dict])
        return self.predict_dataframe(df)[0]

    def predict_dataframe(self, df: pd.DataFrame) -> list:
        """
        Accepts a pandas DataFrame of raw customer records and returns a list of prediction result dicts.
        """
        X_transformed = self.preprocessor.transform(df)
        predictions = self.model.predict(X_transformed)
        probabilities = self.model.predict_proba(X_transformed)[:, 1]

        results = []
        for idx, (pred, prob) in enumerate(zip(predictions, probabilities)):
            prob_percent = float(round(prob * 100, 2))
            risk_level = "Low" if prob < 0.35 else ("Medium" if prob < 0.65 else "High")
            
            results.append({
                'churn_prediction': int(pred),
                'churn_label': "Will Churn" if pred == 1 else "Will Stay",
                'churn_probability': float(round(prob, 4)),
                'churn_probability_percent': prob_percent,
                'risk_level': risk_level
            })

        return results

if __name__ == '__main__':
    # Simple test
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(project_dir, 'models')

    predictor = ChurnPredictor(models_dir=models_dir)
    sample_customer = {
        'CreditScore': 650,
        'Geography': 'Germany',
        'Gender': 'Female',
        'Age': 45,
        'Tenure': 3,
        'Balance': 120000.0,
        'NumOfProducts': 1,
        'HasCrCard': 1,
        'IsActiveMember': 0,
        'EstimatedSalary': 75000.0
    }
    result = predictor.predict_single(sample_customer)
    print("Test Prediction Output:")
    print(result)
