# ==============================================================================
# Stage 1: Build Stage (Builder)
# ==============================================================================
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/models/huggingface \
    PADDLE_HOME=/models/paddle \
    PADDLE_USER_DIR=/models/paddle\    
    SETUPTOOLS_SCM_PRETEND_VERSION=3.5.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip

# 1. PyTorch CPU — largest download, pin the index so it never pulls CUDA wheels
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# 2. PaddlePaddle CPU — second largest, install before paddlex so paddlex
#    sees it already present and doesn't pull its own default wheel
RUN pip install --no-cache-dir "paddlepaddle>=2.6.0"

# 3. opencv-python-headless — replaces opencv-python from requirements.txt;
#    must land before paddlex so paddlex doesn't pull the GUI variant
RUN pip install --no-cache-dir opencv-python-headless

# 4. HuggingFace stack pinned to your floor version — install before paddlex
#    so paddlex cannot downgrade it
RUN pip install --no-cache-dir "transformers>=5.8.0" accelerate tokenizers

# 5. paddlex[ocr] — now sees torch, paddle, opencv, transformers already
#    satisfied and will skip re-downloading them
RUN pip install --no-cache-dir "paddlex[ocr]>=3.5.0,<3.6.0"

# 6. Remaining lightweight deps — everything that isn't already pulled in above.
#    opencv-python is replaced by headless above so we drop it here.
WORKDIR /app
COPY requirements.txt /app/

RUN grep -ivE \
    "^(torch|torchvision|paddlepaddle|paddlex|transformers|opencv-python)" \
    requirements.txt > requirements_remaining.txt && \
    pip install --no-cache-dir -r requirements_remaining.txt

# 7. Install the application package
COPY pyproject.toml setup.py /app/
COPY . /app/
RUN pip install --no-cache-dir .

# ==============================================================================
# Stage 2: Runtime Stage (Runner)
# ==============================================================================
FROM python:3.10-slim AS runner

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000 \
    HF_HOME=/models/huggingface \
    PADDLE_HOME=/models/paddle \
    PADDLE_USER_DIR=/models/paddle \
    PATH="/opt/venv/bin:$PATH" \
    SETUPTOOLS_SCM_PRETEND_VERSION=3.5.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -m -s /bin/bash appuser

RUN mkdir -p /models/huggingface /models/paddle && \
    chown -R appuser:appgroup /models

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY main.py /app/

RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]