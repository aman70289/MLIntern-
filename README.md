# 🚀 End-to-End Customer Churn Machine Learning Pipeline & API

An enterprise-ready machine learning system for predicting customer churn probabilities and risk levels using **Scikit-Learn**, **Joblib**, and **FastAPI**.

---

## 🌟 Key Features
- **Data Preprocessing & Encoding**: Custom `ColumnTransformer` handling continuous scaling (`StandardScaler`) and categorical dummy encoding (`OneHotEncoder`).
- **Model Training & Benchmark**: Multi-model candidate comparison (Logistic Regression, Random Forest, Gradient Boosting) selecting the optimal architecture based on ROC-AUC and F1-score.
- **Artifact Serialization**: Robust model and preprocessor state persistence via **Joblib**.
- **RESTful API**: Production FastAPI service featuring single & batch prediction endpoints, Pydantic input validation, CORS support, and automatic OpenAPI / Swagger documentation.
- **Interactive Web Dashboard**: Embedded dark-mode UI with instant probability calculations, risk level indicators (Low/Medium/High), and business action recommendations.

---

## 📂 Project Architecture

```
ml_churn_project/
├── data/
│   ├── raw_customer_churn.csv          # Generated raw dataset
├── models/
│   ├── preprocessor.joblib             # Serialized ColumnTransformer
│   ├── churn_model.joblib              # Serialized GradientBoosting model
│   └── model_metadata.json             # Metrics & feature schema metadata
├── src/
│   ├── data_preprocessing.py           # Preprocessing & pipeline definition
│   ├── train.py                        # Model training & benchmarking pipeline
│   └── predict.py                      # Production inference class
├── app/
│   ├── main.py                         # FastAPI application server
│   ├── schemas.py                      # Pydantic data schemas
│   └── templates/
│       └── index.html                  # Responsive Web Dashboard UI
├── tests/
│   ├── test_model.py                   # Unit tests for preprocessing & inference
│   └── test_api.py                     # Integration tests for API endpoints
├── generate_dataset.py                 # Dataset generator script
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

---

## 🛠️ Quick Start

### 1. Installation
```bash
git clone https://github.com/your-repo/ml_churn_project.git
cd ml_churn_project
python -m pip install -r requirements.txt
```

### 2. Generate Dataset & Train Model
```bash
python generate_dataset.py
python src/train.py
```

### 3. Run Automated Tests
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 4. Launch Web Dashboard & REST API

#### Option A: FastAPI Web App & Swagger REST API
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# or click run_fastapi.bat
```
- Interactive Dashboard: `http://localhost:8000`
- Swagger API Docs: `http://localhost:8000/docs`

#### Option B: Streamlit Simplified App
```bash
python -m streamlit run app/streamlit_app.py
# or click run_streamlit.bat
```
- App UI: `http://localhost:8501`

#### Option C: Docker Container Deployment
```bash
docker compose up --build
```

---

## 📊 Model Performance Benchmarks

| Model Architecture | Accuracy | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **90.17%** | **0.8239** | **0.9652** | **Selected** |
| Random Forest | 89.33% | 0.7975 | 0.9555 | Candidate |
| Logistic Regression | 87.67% | 0.7798 | 0.9426 | Baseline |

---

## 🔌 API Endpoints Summary

- **`GET /health`**: API status & model readiness.
- **`GET /api/v1/metadata`**: Returns model metrics, name, and feature list.
- **`POST /api/v1/predict`**: Accepts single customer JSON profile and returns churn probability & risk level.
- **`POST /api/v1/predict-batch`**: Accepts list of customer profiles and returns aggregated risk summary.
