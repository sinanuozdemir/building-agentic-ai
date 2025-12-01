FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY twilio_app_simple.py .

# Expose port
EXPOSE 5015

# Run the application
CMD ["uvicorn", "twilio_app_simple:app", "--host", "0.0.0.0", "--port", "5015"]

