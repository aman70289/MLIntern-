import os
import json
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from data_preprocessing import prepare_data

def train_and_evaluate(data_path, models_dir='models'):
    print("Preparing data and fitting preprocessor...")
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(data_path, models_dir=models_dir)

    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    results = {}
    best_model = None
    best_score = -1
    best_model_name = ""

    print("\nTraining and evaluating models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)

        results[name] = {
            'Accuracy': float(round(acc, 4)),
            'Precision': float(round(prec, 4)),
            'Recall': float(round(rec, 4)),
            'F1-Score': float(round(f1, 4)),
            'ROC-AUC': float(round(auc, 4))
        }

        print(f"[{name}] Accuracy: {acc:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {auc:.4f}")

        # Choose best model by ROC-AUC or F1
        if auc > best_score:
            best_score = auc
            best_model = model
            best_model_name = name

    print(f"\nBest Performing Model: {best_model_name} (ROC-AUC: {best_score:.4f})")

    # Serialize best model using Joblib
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'churn_model.joblib')
    joblib.dump(best_model, model_path)
    print(f"Serialized model saved to {model_path}")

    # Save metadata JSON
    metadata = {
        'best_model_name': best_model_name,
        'metrics': results[best_model_name],
        'all_models_metrics': results,
        'feature_count': X_train.shape[1],
        'features': preprocessor.feature_names
    }
    metadata_path = os.path.join(models_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Model metadata saved to {metadata_path}")

    return best_model, metadata

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    raw_csv = os.path.join(project_dir, 'data', 'raw_customer_churn.csv')
    models_dir = os.path.join(project_dir, 'models')
    train_and_evaluate(raw_csv, models_dir=models_dir)
