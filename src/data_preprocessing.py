import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

class ChurnDataPreprocessor:
    """
    Handles data cleaning, feature engineering, categorical encoding,
    numerical scaling, and train/test splitting.
    """
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.preprocessor = None
        self.feature_names = None
        self.numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
        self.categorical_features = ['Geography', 'Gender']
        self.passthrough_features = ['HasCrCard', 'IsActiveMember']

    def build_pipeline(self):
        """
        Creates a ColumnTransformer pipeline for continuous scaling and one-hot encoding.
        """
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), self.categorical_features),
                ('pass', 'passthrough', self.passthrough_features)
            ]
        )
        return self.preprocessor

    def fit_transform(self, df):
        """
        Fits the preprocessor on DataFrame and transforms the features.
        """
        X = df.drop(columns=['CustomerId', 'Exited'], errors='ignore')
        y = df['Exited'] if 'Exited' in df.columns else None

        if self.preprocessor is None:
            self.build_pipeline()

        X_transformed = self.preprocessor.fit_transform(X)

        # Get feature names after transformation
        cat_encoder = self.preprocessor.named_transformers_['cat']
        encoded_cat_names = list(cat_encoder.get_feature_names_out(self.categorical_features))
        self.feature_names = self.numeric_features + encoded_cat_names + self.passthrough_features

        return X_transformed, y

    def transform(self, df):
        """
        Transforms new input data using the fitted preprocessor.
        """
        X = df.drop(columns=['CustomerId', 'Exited'], errors='ignore')
        return self.preprocessor.transform(X)

    def save(self):
        """
        Serializes the preprocessor pipeline to disk using joblib.
        """
        os.makedirs(self.models_dir, exist_ok=True)
        preprocessor_path = os.path.join(self.models_dir, 'preprocessor.joblib')
        joblib.dump(self.preprocessor, preprocessor_path)
        print(f"Preprocessor serialized and saved to {preprocessor_path}")

    def load(self):
        """
        Loads the preprocessor pipeline from disk using joblib.
        """
        preprocessor_path = os.path.join(self.models_dir, 'preprocessor.joblib')
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor artifact not found at {preprocessor_path}")
        self.preprocessor = joblib.load(preprocessor_path)
        return self.preprocessor

def prepare_data(raw_csv_path, test_size=0.2, random_state=42, models_dir='models'):
    df = pd.read_csv(raw_csv_path)
    
    # Simple EDA and sanity check
    df = df.drop_duplicates()
    df = df.dropna()

    preprocessor = ChurnDataPreprocessor(models_dir=models_dir)
    X_transformed, y = preprocessor.fit_transform(df)
    preprocessor.save()

    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, preprocessor
