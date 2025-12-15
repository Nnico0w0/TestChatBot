# Dockerfile for TestChatBot
# Multi-stage build for optimized image size

FROM python:3.10-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# Note: Using --trusted-host flags for environments with SSL certificate issues
RUN pip install --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org \
    -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed models/checkpoints models/tokenizer models/final

# Download NLTK data (required by the application)
# Note: In some CI/CD environments, NLTK downloads may fail due to SSL issues
# The application will attempt to download the data on first run if needed
RUN python download_nltk_data.py

# Expose port for FastAPI
EXPOSE 8000

# Health check - simple check that the server is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

# Default command to run the API
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
