FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire repository
COPY . /app/

# Install the package in development mode
RUN pip install -e .

# Expose the port
EXPOSE 8000

# Run with Gunicorn and Uvicorn workers
CMD ["gunicorn", "app:app", "--workers", "1", "--threads", "8", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]