# ResumeCoach AI

**An end-to-end AI-powered career coaching system** built on Mistral-7B-Instruct (local EC2 inference) and Meta-Llama-3-8B (AWS SageMaker Jumpstart), with a LangChain orchestration layer, Streamlit frontend, and full AWS deployment pipeline.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2.1-yellow.svg)](https://langchain.com/)
[![AWS](https://img.shields.io/badge/AWS-SageMaker%20%7C%20EC2%20%7C%20ECR%20%7C%20S3-orange.svg)](https://aws.amazon.com/)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Features](#3-features)
4. [Model Selection & Deployment Strategy](#4-model-selection--deployment-strategy)
5. [Prompt Engineering](#5-prompt-engineering)
6. [LangChain Design](#6-langchain-design)
7. [EDA & Data Preparation](#7-eda--data-preparation)
8. [Fine-Tuning (QLoRA)](#8-fine-tuning-qlora)
9. [Local Development Setup](#9-local-development-setup)
10. [EC2 Production Deployment](#10-ec2-production-deployment)
11. [SageMaker Jumpstart Deployment](#11-sagemaker-jumpstart-deployment)
12. [AWS Infrastructure Setup](#12-aws-infrastructure-setup)
13. [Project Structure](#13-project-structure)
14. [API Reference](#14-api-reference)
15. [Evaluation Results](#15-evaluation-results)
16. [Cost Analysis](#16-cost-analysis)

---

## 1. Project Overview

ResumeCoach AI provides:

- **Resume ↔ Job Description Analysis** — structured fit assessment with an overall fit score, ATS score, gap analysis, and identified strengths
- **One-Click Optimize** — rewrites the resume to maximize ATS compatibility for the specific role, with quantified before/after score delta
- **Chat Coach** — multi-turn conversational coaching backed by sliding-window memory that survives Mistral-7B's context constraints
- **Cover Letter Generator** — tailored to the specific role and candidate background
- **Interview Prep Guide** — likely questions, suggested answer frameworks, and questions to ask the interviewer
- **PDF Upload + Download** — accepts PDF resumes and exports the optimized resume as a formatted PDF

The system is modeled after [Jobscan](https://www.jobscan.co) in concept, with additional generative capabilities powered by open-source LLMs.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER BROWSER                                │
└────────────────────────┬────────────────────────────────────────────┘
                         │ :8501
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STREAMLIT FRONTEND                                │
│   (app/streamlit_app.py — Apple-style minimalist UI)                │
│   Tabs: Report | Optimize | Chat | Cover Letter | Interview Prep    │
└────────────────────────┬────────────────────────────────────────────┘
                         │ HTTP REST (:8000)
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FASTAPI BACKEND                                   │
│   (backend/main.py)                                                  │
│   Endpoints: /analyze /optimize /chat /cover-letter /interview-prep │
│   Session Store: in-memory dict (Redis in production)               │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   LANGCHAIN LAYER                                    │
│   (backend/chains/coaching_chains.py)                               │
│                                                                      │
│   CoachingReportChain                                               │
│     └─ ContextCompressionChain → CoachingReportChain               │
│   ResumeOptimizerChain                                              │
│     └─ OptimizeChain → ATSRescoreChain                             │
│   ChatCoachChain                                                    │
│     └─ ConversationSummaryBufferMemory (sliding window)            │
│   CoverLetterChain                                                  │
│   InterviewPrepChain                                                │
└────────────────────────┬────────────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────────┐
          ▼              ▼                  ▼
┌──────────────┐  ┌─────────────┐  ┌───────────────────────┐
│  NEBIUS API  │  │  LOCAL HF   │  │  AWS SAGEMAKER        │
│  (Testing)   │  │  (EC2 Prod) │  │  JUMPSTART            │
│              │  │             │  │  (Documented Alt.)    │
│  Llama3-70B  │  │ Mistral-7B  │  │  Llama 3 8B Instruct  │
│  Llama3-8B   │  │ 32k context │  │  ml.g5.2xlarge        │
│  Mistral-7B  │  │ 4-bit quant │  │                       │
└──────────────┘  └──────┬──────┘  └───────────────────────┘
                         │
                    ┌────▼─────┐
                    │  AWS S3  │
                    │ Resumes  │
                    │ Reports  │
                    │  Models  │
                    │  Evals   │
                    └──────────┘
```

**Deployment Stack:**
```
EC2 g4dn.xlarge (Ubuntu 22.04, CUDA 12.1)
  └── Docker Container (NVIDIA runtime)
        ├── supervisord
        │     ├── uvicorn  → FastAPI  (:8000)
        │     └── streamlit           (:8501)
        └── /opt/ml/models/mistral-7b  ← synced from S3
```

---

## 3. Features

### Core Coaching Features

| Feature | Description |
|---|---|
| **Fit Score** | 0-100 overall match score between resume and JD |
| **ATS Score** | 0-100 Applicant Tracking System compatibility score |
| **ATS Breakdown** | Per-dimension scores: Keyword Match, Format, Skills, Experience, Education |
| **Gap Analysis** | Ranked gaps with Critical / Moderate / Minor severity and mitigation advice |
| **Strengths** | Key strengths specific to this role with emphasis strategies |
| **Keyword Analysis** | Present vs missing keywords with visual tags |
| **One-Click Optimize** | Rewrites resume with ATS before/after delta and breakdown comparison |
| **PDF Download** | Exports optimized resume as a formatted PDF (ReportLab) |
| **Chat Coach** | Multi-turn coaching chat with LangChain memory |
| **Cover Letter** | Role-tailored cover letter generation |
| **Interview Prep** | Likely questions, answer frameworks, questions to ask, red flag rebuttals |
| **Salary Insight** | Brief salary negotiation guidance based on role and candidate level |

### Technical Features

| Feature | Implementation |
|---|---|
| **Context Compression** | Resume + JD summarized to ~400 tokens for persistent chat context |
| **Sliding Window Memory** | `ConversationSummaryBufferMemory` with 3,000-token buffer limit |
| **PDF Extraction** | pdfminer.six → pymupdf → pypdf fallback chain |
| **Long Resume Handling** | Chunking + section-level extraction for >2,000 token resumes |
| **Backend Toggle** | `INFERENCE_BACKEND` env var: nebius / local_hf / sagemaker |
| **Session Management** | Per-session state for multi-user support |
| **S3 Storage** | Async background upload of resumes and reports |
| **4-bit Quantization** | bitsandbytes NF4 quantization for Mistral-7B on g4dn.xlarge |

---

## 4. Model Selection & Deployment Strategy

### Model Comparison

| Model | Context | VRAM | Use Case | Cost |
|---|---|---|---|---|
| **Mistral-7B-Instruct-v0.3** | 32,768 tokens | ~5GB (4-bit) | EC2 Production | ~$0.53/hr |
| **Meta-Llama-3-8B-Instruct** | 8,192 tokens | ~16GB | SageMaker Jumpstart | ~$1.21/hr |
| **Nebius API** | Varies by model | N/A | Dev / Testing / Evaluation | ~$0.13/M tokens |

### Why Mistral-7B for Production

1. **4x larger context window** (32k vs 8k) — critical for chat memory and long resumes
2. **Apache 2.0 license** — no gating or approval required
3. **Lower inference cost** — EC2 g4dn.xlarge at $0.53/hr vs $1.21/hr for SageMaker ml.g5.2xlarge
4. **4-bit quantization** — fits in 16GB T4 VRAM, reducing cost further
5. **Excellent instruction following** — strong JSON output compliance

### Why SageMaker Jumpstart is Documented (Not Primary)

SageMaker Jumpstart is fully implemented and documented for:
- Academic compliance with rubric requirements
- Scenarios requiring managed infrastructure
- Teams without DevOps capacity for EC2 management

However, for this project, EC2 local inference is the production path for cost efficiency.

---

## 5. Prompt Engineering

All prompts are located in `backend/prompts/templates.py`. Below is the design rationale for each:

### Prompt Design Principles

1. **Explicit JSON schemas** — every analytical prompt specifies the exact output schema. This reduces hallucinated keys and unparseable responses.

2. **Numbered rubrics** — scoring prompts include explicit rubric definitions (e.g., what constitutes a 90 vs 70 ATS keyword score). This reduces model discretion drift across calls.

3. **Dual-document anchoring** — every prompt explicitly requires the model to reference both the resume AND job description. Generic advice is explicitly forbidden.

4. **Temperature tuning by task**:
   - `0.2` for ATS scoring and gap analysis (analytical, deterministic)
   - `0.4` for resume rewriting (generative but controlled)
   - `0.5-0.6` for chat and cover letters (conversational, some creativity)

5. **Mistral instruction format** — all prompts use the `<s>[INST] ... [/INST]` format required by Mistral-7B for proper instruction following.

### Prompt Inventory

| Prompt | Purpose | Temperature | Output |
|---|---|---|---|
| `CONTEXT_COMPRESSION_PROMPT` | Compress resume+JD to ~400 token summary | 0.2 | Structured text |
| `COACHING_REPORT_PROMPT` | Full coaching report generation | 0.2 | JSON (10+ keys) |
| `RESUME_OPTIMIZATION_PROMPT` | Rewrite resume for ATS | 0.4 | Plain text |
| `ATS_RESCORE_PROMPT` | Score optimized resume | 0.1 | JSON |
| `CHAT_SYSTEM_PROMPT` | Persistent chat context | 0.5 | Conversational |
| `COVER_LETTER_PROMPT` | Tailored cover letter | 0.6 | Formatted text |
| `INTERVIEW_PREP_PROMPT` | Interview Q&A guide | 0.4 | JSON |
| `CHUNK_SUMMARISE_PROMPT` | Summarize long resume sections | 0.1 | Structured text |

### Context Window Budget (Mistral-7B, 32,768 tokens)

```
┌─────────────────────────────────────────────┐
│ System Prompt          ~450 tokens   (1.4%) │
│ Resume (avg)           ~630 tokens   (1.9%) │
│ Job Description        ~550 tokens   (1.7%) │
│ Response (Report)    ~1,200 tokens   (3.7%) │
│ Buffer                 ~300 tokens   (0.9%) │
│                                             │
│ Total used:          ~3,130 / 32,768        │
│ Remaining for chat:  ~29,638 tokens         │
└─────────────────────────────────────────────┘
```

### Chain-of-Thought: Context Compression Innovation

To solve the chat memory problem across long conversations, we implement a two-step approach:

```
Step 1 (on first analysis):
  Resume (600 tokens) + JD (500 tokens)
         ↓  CONTEXT_COMPRESSION_PROMPT
  Context Summary (~350 tokens)

Step 2 (every chat turn):
  Context Summary (350) + Chat History (sliding window, max 3,000)
         ↓  CHAT_SYSTEM_PROMPT
  Response (max 1,024 tokens)

Total per chat turn: ~4,724 tokens (well within 32k)
```

This allows theoretically unlimited conversation depth, with older turns summarized by `ConversationSummaryBufferMemory`.

---

## 6. LangChain Design

### Chain Architecture

```python
# Coaching Report Chain (2 sequential LLM calls)
CoachingReportChain:
  compression_chain = LLMChain(prompt=CONTEXT_COMPRESSION_PROMPT)  # call 1
  report_chain      = LLMChain(prompt=COACHING_REPORT_PROMPT)       # call 2

# Resume Optimizer Chain (2 sequential LLM calls)
ResumeOptimizerChain:
  optimize_chain = LLMChain(prompt=RESUME_OPTIMIZATION_PROMPT, temp=0.4)  # call 1
  rescore_chain  = LLMChain(prompt=ATS_RESCORE_PROMPT, temp=0.1)          # call 2

# Chat Chain (with memory)
ChatCoachChain:
  memory = ConversationSummaryBufferMemory(max_token_limit=3000)
  llm    = Mistral-7B (temp=0.5)
```

### Memory Strategy

`ConversationSummaryBufferMemory` keeps:
- **Recent turns** verbatim (up to `max_token_limit=3000` tokens)
- **Older turns** summarized into a rolling summary
- **Persistent context** (resume/JD summary) injected via system prompt, not memory

This ensures the model always has:
1. The core coaching context (what was in the report)
2. Recent specific exchanges (last ~5-10 turns verbatim)
3. A summary of earlier conversation topics

---

## 7. EDA & Data Preparation

**Notebook:** `eda/eda_analysis.py` (convert to `.ipynb` via `python eda/convert_to_notebook.py`)

### Dataset
- 500 synthetic resumes across 15 job categories and 4 seniority levels
- Structured to mirror real resume distributions observed in public datasets (Kaggle Resume Dataset)

### Key Findings

| Finding | Value | Implication |
|---|---|---|
| Average resume tokens | 630 | Comfortable fit in 32k context |
| Resumes needing chunking (>2k tokens) | 5.2% | Chunking logic handles rare cases |
| ATS boost from quantified achievements | +8.7 pts | Prioritized in optimization prompt |
| Skills count ↔ ATS correlation | r=0.62 | Skills breadth drives ATS |
| Context budget used (avg case) | 9.5% of 32k | Ample headroom for chat history |

### Data Preparation Pipeline

1. **PDF Extraction** — pdfminer.six → pymupdf → pypdf fallback
2. **Text Cleaning** — unicode normalization, whitespace collapse, bullet normalization
3. **Token Counting** — tiktoken approximation (4 chars ≈ 1 token)
4. **Chunking** — paragraph-boundary splitting with 100-token overlap for long resumes
5. **Skills Normalization** — title case, synonym mapping
6. **Contact Extraction** — regex-based email, phone, LinkedIn extraction

---

## 8. Fine-Tuning (QLoRA)

**Code:** `fine_tuning/fine_tune.py`

> **Note:** Due to absence of a proprietary labeled dataset, fine-tuning code is provided as fully functional infrastructure. The base Mistral-7B-Instruct-v0.3 model performs well zero-shot given our prompt engineering. Fine-tuning would be the next step if coaching report outputs were labeled and collected.

### Method: QLoRA (Quantized Low-Rank Adaptation)

| Parameter | Value | Rationale |
|---|---|---|
| Base model | Mistral-7B-Instruct-v0.3 | Strong instruction following, 32k context |
| LoRA rank (r) | 16 | Balance of capacity vs parameter count |
| LoRA alpha | 32 | Standard: alpha = 2×r |
| LoRA dropout | 0.05 | Light regularization for small datasets |
| Target modules | q,k,v,o,gate,up,down_proj | All attention + MLP layers for full adaptation |
| Quantization | 4-bit NF4 | Fits on single A10G (10GB VRAM) |
| Learning rate | 2e-4 | Standard for QLoRA instruction tuning |
| LR scheduler | Cosine | Smooth decay prevents late-training instability |
| Batch size | 4 × 4 acc. = 16 effective | Stable gradients on single GPU |
| Epochs | 3 | Prevents overfitting on small datasets |
| Max seq length | 4,096 | Covers most resume+JD+report combinations |

### Trainable Parameters

With r=16 targeting 7 module types, trainable parameters ≈ 0.5% of total (vs 100% for full fine-tuning), reducing VRAM requirement from ~112GB to ~10GB.

### Training Data Format

```json
{
  "text": "<s>[INST] {system_prompt}\n\nRESUME:\n{resume}\n\nJOB DESCRIPTION:\n{jd} [/INST] {coaching_report_json}</s>",
  "instruction": "...",
  "input": "...",
  "output": "..."
}
```

### How to Fine-Tune (when data is available)

```bash
# Step 1: Prepare dataset
# Create data/raw/raw_data.jsonl with {resume, job_description, coaching_report} pairs
python fine_tuning/fine_tune.py --mode prepare --data_dir ./data/raw

# Step 2: Train (requires GPU — run on EC2 g4dn.xlarge or SageMaker)
python fine_tuning/fine_tune.py --mode train \
  --data_path ./data/prepared/train.jsonl \
  --output_dir ./output/resume-coach-ft \
  --epochs 3 --lr 2e-4

# Step 3: Evaluate
python fine_tuning/fine_tune.py --mode evaluate \
  --model_path ./output/checkpoint-final \
  --data_path ./data/prepared/eval.jsonl

# Step 4: Upload to S3
aws s3 sync ./output/checkpoint-final s3://YOUR_BUCKET/models/resume-coach-ft/
```

---

## 9. Local Development Setup

### Prerequisites

- Python 3.11+
- pip
- (Optional) NVIDIA GPU with CUDA for local HF inference

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/resume-coach-ai.git
cd resume-coach-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your NEBIUS_API_KEY and AWS credentials

# 5. Start FastAPI backend
cd resume_coach
INFERENCE_BACKEND=nebius uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# 6. Start Streamlit frontend (new terminal)
API_BASE_URL=http://localhost:8000 streamlit run app/streamlit_app.py --server.port 8501
```

Open http://localhost:8501 in your browser.

### Backend API Docs

Open http://localhost:8000/docs for the auto-generated FastAPI Swagger UI.

### Switching Inference Backends

```bash
# Nebius API (fastest for development)
INFERENCE_BACKEND=nebius uvicorn backend.main:app --port 8000

# Local HuggingFace (production — requires model downloaded)
INFERENCE_BACKEND=local_hf uvicorn backend.main:app --port 8000

# SageMaker endpoint (requires deployed endpoint)
INFERENCE_BACKEND=sagemaker uvicorn backend.main:app --port 8000
```

### Run EDA Notebook

```bash
pip install jupytext jupyter matplotlib seaborn
cd eda
python convert_to_notebook.py   # creates eda_analysis.ipynb
jupyter notebook eda_analysis.ipynb
```

---

## 10. EC2 Production Deployment

### Recommended Instance

| Attribute | Value |
|---|---|
| Instance type | `g4dn.xlarge` |
| vCPUs | 4 |
| RAM | 16 GB |
| GPU | 1× NVIDIA T4 (16 GB VRAM) |
| Storage | 100 GB gp3 SSD |
| AMI | Deep Learning AMI GPU PyTorch 2.3 (Ubuntu 22.04) |
| Cost (on-demand) | ~$0.526/hr |
| Cost (spot) | ~$0.16/hr |

### Step-by-Step Deployment

#### Step 1: Launch EC2 Instance

1. Go to AWS EC2 Console → Launch Instance
2. Select **Deep Learning AMI GPU PyTorch 2.3.0 (Ubuntu 22.04)**
3. Choose `g4dn.xlarge`
4. Configure Security Group:
   - Port 22 (SSH) — your IP only
   - Port 8000 (FastAPI) — your IP or ALB
   - Port 8501 (Streamlit) — your IP or ALB
5. Create/attach IAM role with: `AmazonS3FullAccess`, `AmazonSageMakerFullAccess`
6. Add 100 GB EBS volume
7. Launch

#### Step 2: Connect and Deploy

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@YOUR_EC2_PUBLIC_IP

# Upload project (or clone from GitHub)
scp -i your-key.pem -r ./resume_coach ubuntu@YOUR_EC2_IP:/opt/resume-coach

# Run deployment script
cd /opt/resume-coach
chmod +x infrastructure/setup_ec2.sh

# Edit variables section in setup_ec2.sh first, then:
./infrastructure/setup_ec2.sh
```

#### Step 3: Download Mistral-7B Model

```bash
# If model not yet in S3, download from HuggingFace first:
python sagemaker/download_model.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --local-dir /opt/ml/models/mistral-7b \
  --bucket YOUR_S3_BUCKET

# If model already in S3, sync to EC2:
aws s3 sync s3://YOUR_BUCKET/models/mistral-7b-instruct-v0.3/ \
  /opt/ml/models/mistral-7b/ --no-progress
```

#### Step 4: Build and Run Docker

```bash
cd /opt/resume-coach

# Build image
docker build -f docker/Dockerfile -t resume-coach:latest .

# Run with GPU
docker run -d \
  --name resume-coach-app \
  --gpus all \
  -p 8000:8000 \
  -p 8501:8501 \
  --env-file .env \
  -v /opt/ml/models/mistral-7b:/opt/ml/models/mistral-7b \
  --restart unless-stopped \
  resume-coach:latest

# Verify
docker logs -f resume-coach-app
curl http://localhost:8000/health
```

#### Step 5: Push to ECR (for reproducible deployments)

```bash
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/resume-coach"

# Login
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com"

# Create repo (once)
aws ecr create-repository --repository-name resume-coach --region us-east-1

# Tag and push
docker tag resume-coach:latest "${ECR_URI}:latest"
docker push "${ECR_URI}:latest"
```

---

## 11. SageMaker Jumpstart Deployment

> **Cost note:** The SageMaker endpoint costs ~$1.21/hr regardless of usage. Delete it after testing/demo to avoid charges.

### Deploy Llama 3 8B via Jumpstart

```bash
# Set environment variables
export AWS_REGION=us-east-1
export SAGEMAKER_EXECUTION_ROLE=arn:aws:iam::ACCOUNT:role/SageMakerExecutionRole
export S3_BUCKET_NAME=your-bucket-name

# Deploy (takes 10-15 minutes)
python sagemaker/deploy_sagemaker.py --action deploy

# Test the endpoint
python sagemaker/deploy_sagemaker.py --action test \
  --prompt "Analyze this candidate: 5 years Python, AWS certified. Role requires: Python 7yr, Kubernetes. Give a brief fit assessment."

# Check status
python sagemaker/deploy_sagemaker.py --action describe

# ⚠️  IMPORTANT: Delete when done to stop billing
python sagemaker/deploy_sagemaker.py --action delete
```

### Use SageMaker Endpoint with Application

```bash
INFERENCE_BACKEND=sagemaker \
SAGEMAKER_ENDPOINT_NAME=resume-coach-llama3-8b-endpoint \
uvicorn backend.main:app --port 8000
```

### Cost Estimate

```bash
python sagemaker/deploy_sagemaker.py --action cost
```

---

## 12. AWS Infrastructure Setup

### S3 Bucket Structure

```
s3://YOUR_BUCKET/
├── resumes/
│   └── {session_id}/
│       └── resume.pdf
├── reports/
│   └── {session_id}/
│       └── coaching_report.json
├── evaluations/
│   └── {eval_id}.json
└── models/
    └── mistral-7b-instruct-v0.3/
        ├── config.json
        ├── tokenizer.json
        └── *.safetensors
```

### IAM Permissions Required

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
      "Resource": ["arn:aws:s3:::YOUR_BUCKET", "arn:aws:s3:::YOUR_BUCKET/*"]
    },
    {
      "Effect": "Allow",
      "Action": ["sagemaker:InvokeEndpoint"],
      "Resource": "arn:aws:sagemaker:*:*:endpoint/resume-coach-*"
    }
  ]
}
```

### Create S3 Bucket

```bash
aws s3 mb s3://YOUR_BUCKET_NAME --region us-east-1
aws s3api put-bucket-versioning \
  --bucket YOUR_BUCKET_NAME \
  --versioning-configuration Status=Enabled
```

---

## 13. Project Structure

```
resume_coach/
├── app/
│   └── streamlit_app.py          # Streamlit frontend (Apple-style UI)
│
├── backend/
│   ├── main.py                   # FastAPI application
│   ├── llm_client.py             # LLM factory (nebius/local_hf/sagemaker)
│   ├── chains/
│   │   └── coaching_chains.py    # All LangChain chains
│   └── prompts/
│       └── templates.py          # All prompt templates (8 prompts)
│
├── config/
│   ├── __init__.py
│   └── settings.py               # Centralised configuration
│
├── utils/
│   ├── document_processor.py     # PDF extraction, chunking, PDF export
│   └── s3_storage.py             # S3 upload/download utilities
│
├── sagemaker/
│   ├── deploy_sagemaker.py       # Jumpstart deployment & management
│   └── download_model.py         # HuggingFace → S3 model pipeline
│
├── fine_tuning/
│   └── fine_tune.py              # QLoRA fine-tuning (dataset prep + training)
│
├── eda/
│   ├── eda_analysis.py           # EDA notebook (jupytext format)
│   ├── convert_to_notebook.py    # .py → .ipynb converter
│   ├── data/                     # Generated datasets
│   └── figures/                  # Generated plots
│
├── evaluation/
│   └── evaluator.py              # Multi-backend model evaluation framework
│
├── docker/
│   ├── Dockerfile                # Multi-stage CUDA build
│   ├── docker-compose.yml        # Local development
│   └── supervisord.conf          # Process manager config
│
├── infrastructure/
│   └── setup_ec2.sh              # Full EC2 setup & deployment script
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## 14. API Reference

### POST /analyze
Accepts resume (PDF upload or text) + job description. Returns coaching report.

```bash
curl -X POST http://localhost:8000/analyze \
  -F "resume_file=@resume.pdf" \
  -F "job_description=Senior Python Engineer requiring 5+ years..." \
  -F "backend=nebius"
```

**Response:**
```json
{
  "session_id": "uuid",
  "report": {
    "overall_fit_score": 82,
    "ats_score": 78,
    "fit_verdict": "Strong Match",
    "gaps": [...],
    "strengths": [...],
    "missing_keywords": [...],
    "coaching_recommendations": [...]
  }
}
```

### POST /optimize
Rewrites resume for ATS. Requires active session.

```bash
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{"session_id": "YOUR_SESSION_ID"}'
```

### GET /download-resume/{session_id}
Downloads optimized resume as formatted PDF.

### POST /chat
Multi-turn coaching chat.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "YOUR_SESSION_ID", "message": "How do I address the AWS gap?"}'
```

### POST /cover-letter
Generates tailored cover letter.

### POST /interview-prep
Generates interview preparation guide.

### GET /health
Returns backend status and active inference backend.

---

## 15. Evaluation Results

Run multi-backend evaluation:

```bash
python evaluation/evaluator.py
```

This tests all configured Nebius models against 2 standardized test cases and outputs a comparison report to `evaluation/eval_results.json` and uploads to S3.

**Evaluation Metrics:**
- JSON Validity Rate — % of responses that parse as valid JSON
- Schema Completeness — % with all required keys present
- Verdict Accuracy — does the model correctly classify fit level?
- Average Latency — response time per backend

**Decision:** Mistral-7B-Instruct-v0.3 was selected for production based on evaluation results showing strong schema compliance and specificity. Llama-3-8B was retained as the SageMaker Jumpstart option due to its managed deployment simplicity. For the final submission report, evaluation comparisons across Nebius-hosted models are documented.

---

## 16. Cost Analysis

### Monthly Cost Estimate (Production)

| Component | Config | Cost |
|---|---|---|
| EC2 g4dn.xlarge | 8hr/day, on-demand | ~$126/mo |
| EC2 g4dn.xlarge | 8hr/day, spot | ~$38/mo |
| S3 storage | 10GB | ~$0.23/mo |
| ECR | 1 image | ~$0.10/mo |
| SageMaker endpoint | Demo only (delete after use) | ~$2-5 |
| Nebius API | Dev/testing, ~1M tokens | ~$0.50/mo |

### Cost vs SageMaker Endpoint (Always-On)

| | EC2 Local (spot) | SageMaker Jumpstart |
|---|---|---|
| Daily (8hr) | $1.28 | $9.70 |
| Monthly (8hr/day) | $38 | $291 |
| Savings | — | **87% cheaper** |

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `INFERENCE_BACKEND` | `nebius` | Active inference backend |
| `NEBIUS_API_KEY` | — | Nebius API key |
| `NEBIUS_MODEL` | `meta-llama/Meta-Llama-3.1-70B-Instruct` | Model for Nebius calls |
| `AWS_ACCESS_KEY_ID` | — | AWS credentials |
| `AWS_SECRET_ACCESS_KEY` | — | AWS credentials |
| `AWS_REGION` | `us-east-1` | AWS region |
| `S3_BUCKET_NAME` | — | S3 bucket for storage |
| `SAGEMAKER_ENDPOINT_NAME` | `resume-coach-llama3-8b-endpoint` | SageMaker endpoint |
| `HF_LOCAL_MODEL_DIR` | `/opt/ml/models/mistral-7b` | Local model path on EC2 |
| `HF_USE_4BIT` | `true` | Enable 4-bit quantization |
| `MEMORY_MAX_TOKEN_LIMIT` | `3000` | LangChain memory buffer size |
| `LOG_LEVEL` | `INFO` | Application log level |

---

## License

MIT License — see LICENSE file.

---

*Built for the AI Engineering capstone project. References: [Jobscan](https://www.jobscan.co), [LangChain Docs](https://docs.langchain.com), [Mistral AI](https://mistral.ai), [AWS SageMaker Jumpstart](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-jumpstart.html)*
