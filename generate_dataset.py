import os
import numpy as np
import pandas as pd

def generate_customer_churn_data(n_samples=2500, random_state=42):
    """
    Generates a realistic synthetic dataset for Bank Customer Churn Prediction.
    """
    np.random.seed(random_state)
    
    customer_ids = np.arange(10001, 10001 + n_samples)
    credit_scores = np.random.randint(350, 850, size=n_samples)
    geographies = np.random.choice(['France', 'Spain', 'Germany'], size=n_samples, p=[0.5, 0.25, 0.25])
    genders = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.54, 0.46])
    ages = np.random.randint(18, 75, size=n_samples)
    tenures = np.random.randint(0, 11, size=n_samples)
    balances = np.round(np.random.uniform(0, 200000, size=n_samples), 2)
    # Some customers have 0 balance
    zero_balance_idx = np.random.choice(n_samples, size=int(n_samples * 0.35), replace=False)
    balances[zero_balance_idx] = 0.0
    
    num_products = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.5, 0.4, 0.07, 0.03])
    has_crcard = np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])
    is_active_member = np.random.choice([0, 1], size=n_samples, p=[0.48, 0.52])
    estimated_salaries = np.round(np.random.uniform(15000, 150000, size=n_samples), 2)
    
    # Calculate churn probability based on features (ground truth logic with noise)
    logit = (
        - 1.5
        + (ages - 40) * 0.075
        - (credit_scores - 600) * 0.003
        + (balances / 100000) * 0.4
        + (geographies == 'Germany') * 0.8
        - (is_active_member == 1) * 0.9
        + (genders == 'Female') * 0.3
        + (num_products == 3) * 1.2
        + (num_products == 4) * 2.5
        - (num_products == 2) * 0.5
        + np.random.normal(0, 0.5, size=n_samples)
    )
    
    probabilities = 1 / (1 + np.exp(-logit))
    churn = (probabilities > 0.5).astype(int)
    
    df = pd.DataFrame({
        'CustomerId': customer_ids,
        'CreditScore': credit_scores,
        'Geography': geographies,
        'Gender': genders,
        'Age': ages,
        'Tenure': tenures,
        'Balance': balances,
        'NumOfProducts': num_products,
        'HasCrCard': has_crcard,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salaries,
        'Exited': churn
    })
    
    return df

if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    df = generate_customer_churn_data(n_samples=3000)
    output_path = os.path.join(data_dir, 'raw_customer_churn.csv')
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created with {len(df)} records at {output_path}")
    print("Class distribution:")
    print(df['Exited'].value_counts(normalize=True))
