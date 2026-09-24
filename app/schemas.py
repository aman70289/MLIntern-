from pydantic import BaseModel, Field
from typing import List, Optional

class CustomerData(BaseModel):
    CreditScore: int = Field(..., ge=300, le=900, example=650, description="Customer credit score (300-900)")
    Geography: str = Field(..., example="France", description="Country: France, Germany, or Spain")
    Gender: str = Field(..., example="Female", description="Gender: Male or Female")
    Age: int = Field(..., ge=18, le=100, example=42, description="Customer age in years")
    Tenure: int = Field(..., ge=0, le=15, example=5, description="Years customer has been with bank")
    Balance: float = Field(..., ge=0.0, example=75000.50, description="Account balance in USD")
    NumOfProducts: int = Field(..., ge=1, le=4, example=2, description="Number of bank products used")
    HasCrCard: int = Field(..., ge=0, le=1, example=1, description="Has Credit Card (1 = Yes, 0 = No)")
    IsActiveMember: int = Field(..., ge=0, le=1, example=1, description="Is Active Member (1 = Yes, 0 = No)")
    EstimatedSalary: float = Field(..., ge=0.0, example=55000.00, description="Estimated annual salary in USD")

class BatchCustomerData(BaseModel):
    customers: List[CustomerData]

class PredictionResult(BaseModel):
    churn_prediction: int
    churn_label: str
    churn_probability: float
    churn_probability_percent: float
    risk_level: str

class BatchPredictionResult(BaseModel):
    total_customers: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    predictions: List[PredictionResult]

class ModelMetadataResponse(BaseModel):
    best_model_name: str
    metrics: dict
    feature_count: int
    features: List[str]
