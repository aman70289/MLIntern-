import unittest
import os
import sys
from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from app.main import app

class TestAPIEndpoints(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["status"], "healthy")

    def test_get_metadata(self):
        response = self.client.get("/api/v1/metadata")
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertIn("best_model_name", json_data)
        self.assertIn("metrics", json_data)

    def test_predict_endpoint(self):
        payload = {
            "CreditScore": 650,
            "Geography": "Germany",
            "Gender": "Female",
            "Age": 55,
            "Tenure": 2,
            "Balance": 120000.0,
            "NumOfProducts": 3,
            "HasCrCard": 1,
            "IsActiveMember": 0,
            "EstimatedSalary": 50000.0
        }
        response = self.client.post("/api/v1/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("churn_prediction", data)
        self.assertIn("churn_probability_percent", data)
        self.assertIn("risk_level", data)

    def test_predict_invalid_data(self):
        # Invalid credit score out of range
        payload = {
            "CreditScore": 100,
            "Geography": "France",
            "Gender": "Male",
            "Age": 25,
            "Tenure": 2,
            "Balance": 1000.0,
            "NumOfProducts": 1,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 20000.0
        }
        response = self.client.post("/api/v1/predict", json=payload)
        self.assertEqual(response.status_code, 422) # Validation error

if __name__ == '__main__':
    unittest.main()
