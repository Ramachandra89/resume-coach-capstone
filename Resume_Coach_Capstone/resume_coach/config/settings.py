"""
Resume Coach - Configuration Settings
Supports three inference backends: nebius | local_hf | sagemaker
"""

import os
from enum import Enum
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


class InferenceBackend(str, Enum):
    NEBIUS = "nebius"
    LOCAL_HF = "local_hf"
    SAGEMAKER = "sagemaker"


# ─────────────────────────────────────────────
# INFERENCE BACKEND TOGGLE
# Set via environment variable INFERENCE_BACKEND
# Options: "nebius" | "local_hf" | "sagemaker"
# ─────────────────────────────────────────────
INFERENCE_BACKEND: InferenceBackend = InferenceBackend(
    os.getenv("INFERENCE_BACKEND", "nebius")
)

# ─────────────────────────────────────────────
# NEBIUS CONFIG (dev / testing / evaluation)
# ─────────────────────────────────────────────
NEBIUS_API_KEY: str = os.getenv("NEBIUS_API_KEY", "YOUR_NEBIUS_API_KEY_HERE")
NEBIUS_BASE_URL: str = "https://api.studio.nebius.ai/v1/"

# Models available on Nebius for evaluation comparisons:
# "meta-llama/Meta-Llama-3-8B-Instruct"
# "meta-llama/Meta-Llama-3.1-70B-Instruct"
# "mistralai/Mistral-7B-Instruct-v0.3"
# "mistralai/Mixtral-8x7B-Instruct-v0.1"
NEBIUS_MODEL: str = os.getenv(
    "NEBIUS_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct"
)

# ─────────────────────────────────────────────
# LOCAL HF CONFIG (EC2 production inference)
# Model: Llama 3.1 8B Instruct
# Context: 128,000 tokens
# Quantization: 4-bit (fits g4dn.xlarge 16GB VRAM)
# Note: Llama 3.1 is a gated model — HF_TOKEN required to download
# ─────────────────────────────────────────────
HF_MODEL_ID: str = os.getenv(
    "HF_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct"
)
HF_MODEL_S3_PATH: str = os.getenv(
    "HF_MODEL_S3_PATH", "s3://YOUR_BUCKET_NAME/models/llama-3.1-8b-instruct/"
)
HF_LOCAL_MODEL_DIR: str = os.getenv("HF_LOCAL_MODEL_DIR", "/opt/ml/models/llama-3.1-8b")
HF_USE_4BIT: bool = os.getenv("HF_USE_4BIT", "true").lower() == "true"
HF_MAX_NEW_TOKENS: int = int(os.getenv("HF_MAX_NEW_TOKENS", "2048"))
HF_TEMPERATURE: float = float(os.getenv("HF_TEMPERATURE", "0.3"))
HF_TOP_P: float = float(os.getenv("HF_TOP_P", "0.9"))

# ─────────────────────────────────────────────
# SAGEMAKER CONFIG (active production backend)
# Model: Llama-3.2-3B-Instruct via JumpStart
# Instance: ml.g5.xlarge (~$1.01/hr)
# ─────────────────────────────────────────────
SAGEMAKER_ENDPOINT_NAME: str = os.getenv(
    "SAGEMAKER_ENDPOINT_NAME", "resume-coach-llama3-8b-endpoint"
)
SAGEMAKER_REGION: str = os.getenv("AWS_REGION", "us-east-1")
SAGEMAKER_MAX_NEW_TOKENS: int = int(os.getenv("SAGEMAKER_MAX_NEW_TOKENS", "1024"))
SAGEMAKER_TEMPERATURE: float = float(os.getenv("SAGEMAKER_TEMPERATURE", "0.3"))

# ─────────────────────────────────────────────
# AWS CONFIG
# ─────────────────────────────────────────────
AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "YOUR_BUCKET_NAME")
S3_RESUMES_PREFIX: str = "resumes/"
S3_REPORTS_PREFIX: str = "reports/"
S3_EVALUATIONS_PREFIX: str = "evaluations/"

# ─────────────────────────────────────────────
# APPLICATION CONFIG
# ─────────────────────────────────────────────
APP_TITLE: str = "ResumeCoach AI"
APP_VERSION: str = "1.0.0"
APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT: int = int(os.getenv("APP_PORT", "8000"))
STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_PORT", "8501"))

# LangChain memory: max tokens to keep in buffer before summarising
MEMORY_MAX_TOKEN_LIMIT: int = int(os.getenv("MEMORY_MAX_TOKEN_LIMIT", "3000"))

# Max resume tokens before chunking + summarisation is triggered
MAX_RESUME_TOKENS: int = int(os.getenv("MAX_RESUME_TOKENS", "2000"))

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
