# Use Python 3.11 (stable, less memory than 3.14)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy dashboard files
COPY . .

# Copy data and model
COPY data/ ./data/
COPY ../src/model.pkl ./model.pkl

# Expose port
EXPOSE 7860

# Run app
CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "1", "--threads", "2"]