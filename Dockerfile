# Use official lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Set working directory
WORKDIR /app

# Install system dependencies needed for OpenCV, PaddleOCR, and compiling cython/etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install paddlepaddle (CPU version)
RUN pip install --no-cache-dir paddlepaddle==3.0.0b2 || pip install --no-cache-dir paddlepaddle

# Copy the entire PaddleOCR project code
COPY . .

# Ensure app package is in the Python search path
ENV PYTHONPATH=/app

# Expose server port
EXPOSE 8000

# Run uvicorn production server
# Replace the last line (CMD) with this — uses Railway's dynamic $PORT
CMD uvicorn api.main:app --host 0.0.0.0 --port $PORT
