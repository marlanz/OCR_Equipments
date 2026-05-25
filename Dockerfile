# ==============================================================================
# Base Build Stage & Runtime Environment
# ==============================================================================
FROM python:3.10-slim

# Set system-level environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000

# Install runtime and compile-time system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create a secure non-root user
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -m -s /bin/bash appuser

# Set working directory and set owner permissions
WORKDIR /app
RUN chown -R appuser:appgroup /app

# Switch to the non-root user
USER appuser
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy dependency files first to leverage Docker layer caching
COPY --chown=appuser:appgroup requirements.txt pyproject.toml setup.py /app/

# Install Python requirements from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY --chown=appuser:appgroup . /app/

# Install the local package in editable/development mode
RUN pip install --no-cache-dir -e .

# Pre-download and cache PaddleOCR-VL-1.5 model weights
# This saves VLM models into /home/appuser/.cache and /home/appuser/.paddleocr
RUN python -c "from paddleocr import PaddleOCRVL; PaddleOCRVL(pipeline_version='v1.5', engine='transformers', device='cpu', use_doc_orientation_classify=False, use_doc_unwarping=False, use_layout_detection=True)"

# Expose the application port
EXPOSE 8000

# Run the FastAPI application using Uvicorn
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
