# Use official lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV PYTHONPATH=/app
ENV SETUPTOOLS_SCM_PRETEND_VERSION=3.0.0

# Set working directory
WORKDIR /app

# Install minimal system dependencies (excluding bulky GUI/X11 libraries since we use headless OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install core Python requirements first (leverages Docker cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Step 1: Pre-install CPU PyTorch explicitly (prevents GPU bloat from paddlex)
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Step 2: Pre-install CPU PaddlePaddle explicitly from official Baidu CPU repository mirror
RUN pip install --no-cache-dir \
    paddlepaddle==3.1.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# Step 3: Pre-install PaddleX from the correct tested release branch (matches repository requirements)
RUN pip install --no-cache-dir \
    "paddlex@git+https://github.com/PaddlePaddle/PaddleX.git@release/3.5"

# Step 4: Pre-install Hugging Face Transformers & Hub for PaddleOCRVL VLM pipelines
RUN pip install --no-cache-dir \
    "transformers>=5.8.0" \
    huggingface_hub

# Step 5: Copy the entire local project codebase
COPY . .

# Step 6: Install the local PaddleOCR package in local/editable mode to bind the codebase
RUN pip install --no-cache-dir -e .

# Expose container port
EXPOSE 8000

# Start the API - Railway injects the environment variable $PORT
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]