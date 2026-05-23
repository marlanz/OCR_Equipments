# Use official lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install base dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install paddlepaddle CPU explicitly FIRST before anything else pulls it in
RUN pip install --no-cache-dir paddlepaddle==3.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple \
    || pip install --no-cache-dir paddlepaddle==3.0.0 --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cpu/ \
    || pip install --no-cache-dir paddlepaddle

# Install PaddleOCR stack — use --no-deps on paddlex to prevent it from pulling CUDA torch
RUN pip install --no-cache-dir "paddlex[ocr]>=3.5.0,<3.6.0" paddleocr>=2.9.0 transformers==5.9.0 huggingface_hub

# Force CPU torch LAST so paddlex cannot overwrite it
RUN pip install --no-cache-dir --force-reinstall \
    torch==2.4.0 \
    torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Copy the entire project code
COPY . .

# Ensure app package is in the Python search path
ENV PYTHONPATH=/app

# Expose server port
EXPOSE 8000

# Run uvicorn — $PORT is injected by Railway at runtime
CMD uvicorn api.main:app --host 0.0.0.0 --port $PORT