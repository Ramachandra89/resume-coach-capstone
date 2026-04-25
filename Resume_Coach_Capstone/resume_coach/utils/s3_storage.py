"""
Resume Coach - S3 Storage Utilities
Handles upload/download of resumes, reports, and evaluation data.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def get_s3_client():
    import boto3
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from config.settings import AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        return boto3.client(
            's3',
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
    # Use IAM role if running on EC2
    return boto3.client('s3', region_name=AWS_REGION)


def upload_resume(
    file_bytes: bytes,
    filename: str,
    session_id: str,
    bucket: Optional[str] = None
) -> str:
    """Upload resume PDF to S3. Returns S3 key."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from config.settings import S3_BUCKET_NAME, S3_RESUMES_PREFIX

    bucket = bucket or S3_BUCKET_NAME
    key = f"{S3_RESUMES_PREFIX}{session_id}/{filename}"

    try:
        s3 = get_s3_client()
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=file_bytes,
            ContentType='application/pdf',
            Metadata={'session_id': session_id, 'uploaded_at': datetime.utcnow().isoformat()},
        )
        logger.info(f"Resume uploaded to s3://{bucket}/{key}")
        return key
    except Exception as e:
        logger.error(f"Failed to upload resume to S3: {e}")
        raise


def upload_report(
    report_data: Dict[str, Any],
    session_id: str,
    bucket: Optional[str] = None
) -> str:
    """Upload coaching report JSON to S3. Returns S3 key."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from config.settings import S3_BUCKET_NAME, S3_REPORTS_PREFIX

    bucket = bucket or S3_BUCKET_NAME
    key = f"{S3_REPORTS_PREFIX}{session_id}/coaching_report.json"

    try:
        s3 = get_s3_client()
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(report_data, indent=2),
            ContentType='application/json',
        )
        logger.info(f"Report uploaded to s3://{bucket}/{key}")
        return key
    except Exception as e:
        logger.error(f"Failed to upload report to S3: {e}")
        raise


def upload_evaluation(
    evaluation_data: Dict[str, Any],
    eval_id: Optional[str] = None,
    bucket: Optional[str] = None,
) -> str:
    """Upload model evaluation results to S3."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from config.settings import S3_BUCKET_NAME, S3_EVALUATIONS_PREFIX

    bucket = bucket or S3_BUCKET_NAME
    eval_id = eval_id or str(uuid.uuid4())
    key = f"{S3_EVALUATIONS_PREFIX}{eval_id}.json"

    try:
        s3 = get_s3_client()
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(evaluation_data, indent=2),
            ContentType='application/json',
        )
        logger.info(f"Evaluation uploaded to s3://{bucket}/{key}")
        return key
    except Exception as e:
        logger.error(f"Failed to upload evaluation to S3: {e}")
        raise


def download_file(key: str, bucket: Optional[str] = None) -> bytes:
    """Download any file from S3 by key."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from config.settings import S3_BUCKET_NAME

    bucket = bucket or S3_BUCKET_NAME
    s3 = get_s3_client()
    response = s3.get_object(Bucket=bucket, Key=key)
    return response['Body'].read()
