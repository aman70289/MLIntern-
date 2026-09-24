@echo off
echo Starting Customer Churn FastAPI Server & Dashboard...
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
pause
