FROM python:3.11-slim

WORKDIR /app

# Copy requirement definition
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Generate dataset and train model on container build if missing
RUN python generate_dataset.py && python src/train.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
