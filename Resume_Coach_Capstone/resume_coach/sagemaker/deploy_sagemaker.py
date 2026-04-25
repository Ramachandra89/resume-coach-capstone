"""
Resume Coach - SageMaker JumpStart Deployment
Deploys meta-llama/Llama-3.2-3B-Instruct via AWS SageMaker JumpStart.

Prerequisites:
  - AWS credentials with SageMaker full access
  - SageMaker execution role with S3 access
  - Llama 3.2 license accepted on HuggingFace

Instance selection rationale:
  - ml.g5.xlarge: 1x NVIDIA A10G (24GB VRAM), ~$1.006/hr
  - Sufficient VRAM for Llama 3.2 3B in FP16 (~6GB)
  - Cost-effective for intermittent use; delete endpoint when not needed

Usage:
  python deploy_sagemaker.py --action deploy
  python deploy_sagemaker.py --action test --prompt "Analyze this resume..."
  python deploy_sagemaker.py --action delete
"""

import argparse
import json
import logging
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
from config.settings import (
    SAGEMAKER_ENDPOINT_NAME,
    SAGEMAKER_REGION,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
)

# SageMaker JumpStart model ID for Llama 3.2 3B Instruct
# Find current IDs at: https://sagemaker.readthedocs.io/en/stable/doc_utils/jumpstart.html
JUMPSTART_MODEL_ID = "meta-textgeneration-llama-3-2-3b-instruct"
JUMPSTART_MODEL_VERSION = "*"  # latest

INSTANCE_TYPE = "ml.trn1.2xlarge"  # AWS Trainium chip — cost-efficient for inference
ENDPOINT_NAME = SAGEMAKER_ENDPOINT_NAME
REGION = SAGEMAKER_REGION


def _get_execution_role() -> str:
    """
    Resolve IAM execution role.
    Inside SageMaker Studio/Notebook: auto-detected from instance metadata.
    Outside SageMaker (local): falls back to SAGEMAKER_EXECUTION_ROLE env var.
    """
    try:
        import sagemaker
        role = sagemaker.get_execution_role()
        logger.info(f"Execution role auto-detected: {role}")
        return role
    except Exception:
        role = os.getenv("SAGEMAKER_EXECUTION_ROLE", "")
        if not role or "YOUR_ACCOUNT_ID" in role:
            raise RuntimeError(
                "Could not auto-detect SageMaker execution role.\n"
                "Set SAGEMAKER_EXECUTION_ROLE=<role-arn> in .env when deploying locally."
            )
        logger.info(f"Execution role from env: {role}")
        return role


# ─────────────────────────────────────────────
# DEPLOYMENT
# ─────────────────────────────────────────────

def deploy_endpoint():
    """
    Deploy Llama 3.2 3B Instruct via SageMaker JumpStart.
    Estimated deployment time: 8-12 minutes.
    """
    try:
        import sagemaker
        from sagemaker.jumpstart.model import JumpStartModel
    except ImportError:
        raise ImportError("Install sagemaker SDK: pip install sagemaker")

    logger.info(f"Starting deployment of {JUMPSTART_MODEL_ID}")
    logger.info(f"Instance: {INSTANCE_TYPE} | Region: {REGION} | Endpoint: {ENDPOINT_NAME}")
    logger.warning("⚠️  Cost reminder: ml.trn1.2xlarge costs money. Delete endpoint when done!")

    try:
        boto_session = _get_boto_session()
        sagemaker_session = sagemaker.Session(
            boto_session=boto_session,
            default_bucket=os.getenv("S3_BUCKET_NAME", "resume-coach-bucket"),
        )

        model = JumpStartModel(
            model_id=JUMPSTART_MODEL_ID,
            model_version=JUMPSTART_MODEL_VERSION,
            role=_get_execution_role(),
            sagemaker_session=sagemaker_session,
            region_name=REGION,
        )

        logger.info("Deploying model... (this takes 8-12 minutes)")
        start_time = time.time()

        predictor = model.deploy(
            initial_instance_count=1,
            instance_type=INSTANCE_TYPE,
            endpoint_name=ENDPOINT_NAME,
            accept_eula=True,  # Required for Llama 3
        )

        elapsed = (time.time() - start_time) / 60
        logger.info(f"✅ Endpoint deployed successfully in {elapsed:.1f} minutes!")
        logger.info(f"Endpoint name: {ENDPOINT_NAME}")
        logger.info("Remember to delete the endpoint when not in use to avoid charges.")

        return predictor

    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise


# ─────────────────────────────────────────────
# INFERENCE TEST
# ─────────────────────────────────────────────

def test_endpoint(prompt: str = None):
    """Test the deployed endpoint with a sample prompt."""
    import boto3

    if not prompt:
        prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a resume coach.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "A candidate has 3 years of Python experience and is applying for a Senior Data Scientist role requiring 5+ years. "
            "Give them 2 specific pieces of advice.<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    logger.info(f"Testing endpoint: {ENDPOINT_NAME}")

    client = _get_boto_session().client("sagemaker-runtime", region_name=REGION)

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.3,
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False,
        },
    }

    try:
        response = client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(payload),
        )
        result = json.loads(response["Body"].read().decode("utf-8"))
        if isinstance(result, list):
            generated = result[0].get("generated_text", "")
        else:
            generated = result.get("generated_text", str(result))

        logger.info("✅ Endpoint test successful!")
        print("\n" + "="*60)
        print("PROMPT:", prompt[:100] + "...")
        print("-"*60)
        print("RESPONSE:", generated)
        print("="*60)
        return generated

    except Exception as e:
        logger.error(f"Endpoint test failed: {e}")
        raise


# ─────────────────────────────────────────────
# DELETE ENDPOINT
# ─────────────────────────────────────────────

def delete_endpoint():
    """
    Delete the SageMaker endpoint to stop incurring charges.
    ⚠️  This is irreversible — you'll need to redeploy to use it again.
    """
    import boto3

    confirm = input(f"Are you sure you want to delete endpoint '{ENDPOINT_NAME}'? (yes/no): ")
    if confirm.lower() != "yes":
        logger.info("Deletion cancelled.")
        return

    client = _get_boto_session().client("sagemaker", region_name=REGION)
    try:
        client.delete_endpoint(EndpointName=ENDPOINT_NAME)
        logger.info(f"✅ Endpoint '{ENDPOINT_NAME}' deleted. No more charges for this endpoint.")
    except client.exceptions.ClientError as e:
        if "Could not find endpoint" in str(e):
            logger.warning("Endpoint not found — may already be deleted.")
        else:
            raise


# ─────────────────────────────────────────────
# DESCRIBE ENDPOINT
# ─────────────────────────────────────────────

def describe_endpoint():
    """Print current endpoint status and configuration."""
    client = _get_boto_session().client("sagemaker", region_name=REGION)
    try:
        info = client.describe_endpoint(EndpointName=ENDPOINT_NAME)
        print(f"\nEndpoint: {info['EndpointName']}")
        print(f"Status: {info['EndpointStatus']}")
        print(f"Created: {info['CreationTime']}")
        print(f"Last modified: {info['LastModifiedTime']}")
    except Exception as e:
        logger.error(f"Could not describe endpoint: {e}")


# ─────────────────────────────────────────────
# COST ESTIMATE
# ─────────────────────────────────────────────

def print_cost_estimate():
    print("""
╔══════════════════════════════════════════════════════╗
║         SageMaker Endpoint Cost Estimate             ║
╠══════════════════════════════════════════════════════╣
║  Model: Llama 3.2 3B Instruct (JumpStart)           ║
║  Instance: ml.trn1.2xlarge (AWS Trainium)           ║
║                                                      ║
║  💡 Cost-saving tips:                               ║
║  - Delete endpoint after testing                     ║
║  - Use Nebius API for development                    ║
║  - Only deploy SageMaker for demos/evaluation        ║
╚══════════════════════════════════════════════════════╝
    """)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _get_boto_session():
    import boto3
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        return boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=REGION,
        )
    return boto3.Session(region_name=REGION)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SageMaker endpoint management")
    parser.add_argument(
        "--action",
        choices=["deploy", "test", "delete", "describe", "cost"],
        required=True,
        help="Action to perform",
    )
    parser.add_argument("--prompt", type=str, help="Custom prompt for testing")
    args = parser.parse_args()

    if args.action == "deploy":
        print_cost_estimate()
        deploy_endpoint()
    elif args.action == "test":
        test_endpoint(args.prompt)
    elif args.action == "delete":
        delete_endpoint()
    elif args.action == "describe":
        describe_endpoint()
    elif args.action == "cost":
        print_cost_estimate()
