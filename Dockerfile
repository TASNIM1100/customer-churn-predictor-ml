# Dockerfile — Churn Predictor
# Build:  docker build -t churn-predictor .
# Run:    docker run -p 8000:8000 -v "%cd%\models:/app/models" churn-predictor

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p models data

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
