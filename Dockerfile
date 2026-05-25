# ==============================================================================
# Stage 1: Build Stage (Builder)
# ==============================================================================
FROM python:3.10-slim AS builder

# Set system-level build environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/models/huggingface \
    PADDLE_HOME=/models/paddle \
    PADDLE_USER_DIR=/models/paddle

# Install build-time system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install CPU-only PyTorch and torchvision
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Set working directory and copy dependency descriptors
WORKDIR /app
COPY requirements.txt pyproject.toml setup.py /app/

# Install application dependencies in the virtual environment
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and install the package
COPY . /app/
RUN pip install --no-cache-dir .

# Create models directory and pre-download VLM models during build
RUN mkdir -p /models && \
    python -c "from paddleocr import PaddleOCRVL; PaddleOCRVL(pipeline_version='v1.5', engine='transformers', device='cpu', use_doc_orientation_classify=False, use_doc_unwarping=False, use_layout_detection=True)"

# ==============================================================================
# Stage 2: Runtime Stage (Runner)
# ==============================================================================
FROM python:3.10-slim AS runner

# Set system-level runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000 \
    HF_HOME=/models/huggingface \
    PADDLE_HOME=/models/paddle \
    PADDLE_USER_DIR=/models/paddle \
    PATH="/opt/venv/bin:$PATH"\
    PORT=8000\
    SETUPTOOLS_SCM_PRETEND_VERSION=3.5.0

# Install ONLY minimal runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create secure non-root user and group
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -m -s /bin/bash appuser

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy pre-downloaded model weights from builder stage
COPY --from=builder /models /models

# Set working directory
WORKDIR /app

# Copy FastAPI entrypoint
COPY main.py /app/

# Setup ownership for application and model directories
RUN chown -R appuser:appgroup /app /models

# Switch to the non-root user
USER appuser

# Expose the application port
EXPOSE 8000

# Run the FastAPI application using Uvicorn
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
