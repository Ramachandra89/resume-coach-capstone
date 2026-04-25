# ─────────────────────────────────────────────────────────────────
# Resume Coach AI - Production Dockerfile
# Multi-stage build: builder + runtime
# Base: CUDA 12.1 for GPU inference on EC2 g4dn.xlarge
# ─────────────────────────────────────────────────────────────────

# Stage 1: Builder — install dependencies
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ─────────────────────────────────────────────────────────────────
# Stage 2: Runtime
# ─────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    curl \
    awscli \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# App directory
WORKDIR /app

# Copy application code
COPY config/ ./config/
COPY backend/ ./backend/
COPY utils/ ./utils/
COPY app/ ./app/
COPY sagemaker/ ./sagemaker/
COPY evaluation/ ./evaluation/

# Create model directory (populated at runtime from S3)
RUN mkdir -p /opt/ml/models/mistral-7b

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app /opt/ml
USER appuser

# Environment defaults (override via docker-compose or EC2 user-data)
ENV INFERENCE_BACKEND=local_hf
ENV APP_HOST=0.0.0.0
ENV APP_PORT=8000
ENV STREAMLIT_PORT=8501
ENV HF_LOCAL_MODEL_DIR=/opt/ml/models/mistral-7b
ENV LOG_LEVEL=INFO

# Expose ports
EXPOSE 8000 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: run both FastAPI and Streamlit via supervisor
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
