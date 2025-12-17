FROM python:3.10-slim

WORKDIR /app

# Install dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Start FastAPI (Cloud Run compatible)
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
