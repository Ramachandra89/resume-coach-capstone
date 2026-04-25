#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# Resume Coach AI - EC2 Setup & Deployment Script
# 
# Recommended EC2 instance: g4dn.xlarge
#   - 4 vCPUs, 16GB RAM, 1x NVIDIA T4 (16GB VRAM)
#   - On-demand: ~$0.526/hr | Spot: ~$0.16/hr
#   - AMI: Deep Learning AMI GPU PyTorch (Ubuntu 20.04)
#     (comes with CUDA, Docker, and NVIDIA drivers pre-installed)
#
# Usage:
#   chmod +x setup_ec2.sh
#   ./setup_ec2.sh
#
# Or as EC2 user-data (paste entire script):
#   Modify VARIABLES section with your values before running.
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

# ─────────────────────────────────────────────
# VARIABLES — Update these before running
# ─────────────────────────────────────────────
export NEBIUS_API_KEY="YOUR_NEBIUS_API_KEY"
export AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_KEY"
export AWS_REGION="us-east-1"
export S3_BUCKET_NAME="YOUR_S3_BUCKET_NAME"
export INFERENCE_BACKEND="local_hf"   # or "nebius" for API-based

ECR_REPO="resume-coach"
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_TAG="${ECR_REGISTRY}/${ECR_REPO}:latest"

APP_DIR="/opt/resume-coach"
MODEL_DIR="/opt/ml/models/mistral-7b"

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
log() { echo "[$(date +'%Y-%m-%dT%H:%M:%S')] $*"; }

# ─────────────────────────────────────────────
# STEP 1: System Updates
# ─────────────────────────────────────────────
log "Step 1: Updating system packages..."
sudo apt-get update -qq
sudo apt-get upgrade -y -qq
sudo apt-get install -y --no-install-recommends \
    git curl wget unzip jq htop nvtop supervisor awscli

# ─────────────────────────────────────────────
# STEP 2: Docker (skip if Deep Learning AMI — already installed)
# ─────────────────────────────────────────────
log "Step 2: Checking Docker..."
if ! command -v docker &> /dev/null; then
    log "Installing Docker..."
    curl -fsSL https://get.docker.com | bash
    sudo usermod -aG docker ubuntu
    sudo systemctl enable docker
    sudo systemctl start docker
else
    log "Docker already installed: $(docker --version)"
fi

# NVIDIA Container Toolkit (for GPU passthrough to Docker)
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    log "Installing NVIDIA Container Toolkit..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L "https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list" \
        | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update -qq
    sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker
fi

# ─────────────────────────────────────────────
# STEP 3: Configure AWS CLI
# ─────────────────────────────────────────────
log "Step 3: Configuring AWS CLI..."
aws configure set aws_access_key_id "${AWS_ACCESS_KEY_ID}"
aws configure set aws_secret_access_key "${AWS_SECRET_ACCESS_KEY}"
aws configure set default.region "${AWS_REGION}"

# Verify AWS access
aws sts get-caller-identity || { log "ERROR: AWS credentials invalid"; exit 1; }

# ─────────────────────────────────────────────
# STEP 4: Create S3 Bucket
# ─────────────────────────────────────────────
log "Step 4: Creating S3 bucket (if not exists)..."
aws s3 mb "s3://${S3_BUCKET_NAME}" --region "${AWS_REGION}" 2>/dev/null || \
    log "Bucket already exists or belongs to another account."

# Create directory structure in S3
for prefix in resumes/ reports/ evaluations/ models/; do
    aws s3api put-object \
        --bucket "${S3_BUCKET_NAME}" \
        --key "${prefix}" 2>/dev/null || true
done

# ─────────────────────────────────────────────
# STEP 5: Clone Application
# ─────────────────────────────────────────────
log "Step 5: Setting up application directory..."
sudo mkdir -p "${APP_DIR}"
sudo chown -R ubuntu:ubuntu "${APP_DIR}"

# If running from git:
# git clone https://github.com/YOUR_USERNAME/resume-coach.git "${APP_DIR}"
# For now, assume code is already on instance:
cp -r . "${APP_DIR}/" 2>/dev/null || true

# ─────────────────────────────────────────────
# STEP 6: Create .env File
# ─────────────────────────────────────────────
log "Step 6: Creating environment file..."
cat > "${APP_DIR}/.env" << EOF
INFERENCE_BACKEND=${INFERENCE_BACKEND}
NEBIUS_API_KEY=${NEBIUS_API_KEY}
NEBIUS_MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct
AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
AWS_REGION=${AWS_REGION}
S3_BUCKET_NAME=${S3_BUCKET_NAME}
HF_LOCAL_MODEL_DIR=${MODEL_DIR}
SAGEMAKER_ENDPOINT_NAME=resume-coach-llama3-8b-endpoint
LOG_LEVEL=INFO
EOF
chmod 600 "${APP_DIR}/.env"

# ─────────────────────────────────────────────
# STEP 7: Create ECR Repository
# ─────────────────────────────────────────────
log "Step 7: Creating ECR repository..."
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_TAG="${ECR_REGISTRY}/${ECR_REPO}:latest"

aws ecr describe-repositories --repository-names "${ECR_REPO}" 2>/dev/null || \
    aws ecr create-repository \
        --repository-name "${ECR_REPO}" \
        --image-scanning-configuration scanOnPush=true \
        --region "${AWS_REGION}"

# ─────────────────────────────────────────────
# STEP 8: Build & Push Docker Image
# ─────────────────────────────────────────────
log "Step 8: Building Docker image..."
cd "${APP_DIR}"

# Login to ECR
aws ecr get-login-password --region "${AWS_REGION}" | \
    docker login --username AWS --password-stdin "${ECR_REGISTRY}"

# Build
docker build -f docker/Dockerfile -t "${ECR_REPO}:latest" .
docker tag "${ECR_REPO}:latest" "${IMAGE_TAG}"

# Push to ECR
log "Pushing image to ECR..."
docker push "${IMAGE_TAG}"
log "✅ Image pushed to ${IMAGE_TAG}"

# ─────────────────────────────────────────────
# STEP 9: Download Model from S3 (if exists)
# ─────────────────────────────────────────────
log "Step 9: Checking for model in S3..."
sudo mkdir -p "${MODEL_DIR}"
sudo chown -R ubuntu:ubuntu /opt/ml

MODEL_S3_URI="s3://${S3_BUCKET_NAME}/models/mistral-7b-instruct-v0.3/"
if aws s3 ls "${MODEL_S3_URI}" 2>/dev/null | head -1 | grep -q .; then
    log "Model found in S3. Downloading... (this takes ~5-10 minutes)"
    aws s3 sync "${MODEL_S3_URI}" "${MODEL_DIR}/" --no-progress
    log "✅ Model downloaded to ${MODEL_DIR}"
else
    log "⚠️  Model not found in S3."
    log "   If using local_hf backend, run: python sagemaker/download_model.py"
    log "   Or set INFERENCE_BACKEND=nebius to use API inference"
fi

# ─────────────────────────────────────────────
# STEP 10: Run Application
# ─────────────────────────────────────────────
log "Step 10: Starting application..."
cd "${APP_DIR}"

# Pull latest image from ECR
docker pull "${IMAGE_TAG}" 2>/dev/null || true

# Stop existing container
docker stop resume-coach-app 2>/dev/null || true
docker rm resume-coach-app 2>/dev/null || true

# Run with GPU support
docker run -d \
    --name resume-coach-app \
    --gpus all \
    -p 8000:8000 \
    -p 8501:8501 \
    --env-file "${APP_DIR}/.env" \
    -v "${MODEL_DIR}:/opt/ml/models/mistral-7b" \
    --restart unless-stopped \
    "${IMAGE_TAG}"

log "✅ Container started!"

# ─────────────────────────────────────────────
# STEP 11: Verify
# ─────────────────────────────────────────────
log "Step 11: Waiting for application to start (60 seconds)..."
sleep 60

if curl -s http://localhost:8000/health | grep -q "healthy"; then
    log "✅ FastAPI backend is healthy!"
else
    log "⚠️  Backend health check failed. Check logs: docker logs resume-coach-app"
fi

# ─────────────────────────────────────────────
# SECURITY: Configure firewall
# ─────────────────────────────────────────────
log "Configuring UFW firewall..."
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 8000/tcp  # FastAPI (internal only — use ALB in production)
sudo ufw allow 8501/tcp  # Streamlit
sudo ufw --force enable

# ─────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────
PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
log ""
log "═══════════════════════════════════════════"
log "✅ ResumeCoach AI Deployment Complete!"
log "═══════════════════════════════════════════"
log ""
log "Application URLs:"
log "  Streamlit UI:  http://${PUBLIC_IP}:8501"
log "  FastAPI Docs:  http://${PUBLIC_IP}:8000/docs"
log "  Health Check:  http://${PUBLIC_IP}:8000/health"
log ""
log "Management commands:"
log "  docker logs -f resume-coach-app    # View logs"
log "  docker restart resume-coach-app    # Restart"
log "  docker stats resume-coach-app      # Resource usage"
log ""
log "⚠️  Remember to configure Security Group to allow ports 8000, 8501"
