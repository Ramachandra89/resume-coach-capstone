"""
Resume Coach - LLM Client Factory
Provides a unified LangChain LLM interface regardless of backend.
"""

import json
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


def get_llm(backend: Optional[str] = None, temperature: float = 0.3, max_tokens: int = 2048):
    """
    Factory function returning a LangChain-compatible LLM.
    Backend priority: argument > env var INFERENCE_BACKEND > "nebius"
    """
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from config.settings import (
        INFERENCE_BACKEND, InferenceBackend,
        NEBIUS_API_KEY, NEBIUS_BASE_URL, NEBIUS_MODEL,
        HF_LOCAL_MODEL_DIR, HF_MODEL_ID, HF_USE_4BIT,
        SAGEMAKER_ENDPOINT_NAME, SAGEMAKER_REGION,
    )

    chosen = backend or INFERENCE_BACKEND

    if chosen == InferenceBackend.NEBIUS or chosen == "nebius":
        return _get_nebius_llm(
            api_key=NEBIUS_API_KEY,
            base_url=NEBIUS_BASE_URL,
            model=NEBIUS_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif chosen == InferenceBackend.LOCAL_HF or chosen == "local_hf":
        return _get_local_hf_llm(
            model_dir=HF_LOCAL_MODEL_DIR,
            model_id=HF_MODEL_ID,
            use_4bit=HF_USE_4BIT,
            temperature=temperature,
            max_new_tokens=max_tokens,
        )
    elif chosen == InferenceBackend.SAGEMAKER or chosen == "sagemaker":
        return _get_sagemaker_llm(
            endpoint_name=SAGEMAKER_ENDPOINT_NAME,
            region=SAGEMAKER_REGION,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        raise ValueError(f"Unknown inference backend: {chosen}")


def _get_nebius_llm(api_key, base_url, model, temperature, max_tokens):
    """OpenAI-compatible client pointing to Nebius inference API."""
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            openai_api_key=api_key,
            openai_api_base=base_url,
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        logger.info(f"Nebius LLM initialised: {model}")
        return llm
    except ImportError:
        raise ImportError("Install langchain-openai: pip install langchain-openai")


def _get_local_hf_llm(model_dir, model_id, use_4bit, temperature, max_new_tokens):
    """
    Loads Llama-3.1-8B-Instruct locally using HuggingFace transformers.
    Uses 4-bit quantization via bitsandbytes to fit on g4dn.xlarge (16GB VRAM).
    Falls back to CPU if no GPU is available (slow, for testing only).
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
        from langchain_community.llms import HuggingFacePipeline

        # Check if local model dir exists; if not, try loading from HF Hub
        import os
        load_path = model_dir if os.path.exists(model_dir) else model_id
        logger.info(f"Loading local HF model from: {load_path}")

        tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)

        if use_4bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                load_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            logger.info("Model loaded with 4-bit quantization (CUDA)")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                load_path,
                torch_dtype=torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )
            logger.warning("4-bit quantization disabled or no CUDA. Using float32.")

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        logger.info("Local HuggingFace LLM initialised (Llama-3.1-8B-Instruct)")
        return llm

    except ImportError as e:
        raise ImportError(
            f"Missing dependency for local HF inference: {e}\n"
            "Install: pip install transformers bitsandbytes accelerate langchain-community"
        )


def _get_sagemaker_llm(endpoint_name, region, temperature, max_tokens):
    """
    Calls a deployed SageMaker JumpStart endpoint (Llama 3.2 3B Instruct).
    Uses langchain_aws SagemakerEndpoint with a custom content handler.
    Prompts must be pre-formatted with Llama 3 special tokens (<|begin_of_text|> etc.).
    """
    try:
        from langchain_aws import SagemakerEndpoint
        from langchain_aws.llms.sagemaker_endpoint import LLMContentHandler

        class Llama32ContentHandler(LLMContentHandler):
            content_type = "application/json"
            accepts = "application/json"

            def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": model_kwargs.get("max_new_tokens", max_tokens),
                        "temperature": model_kwargs.get("temperature", temperature),
                        "top_p": 0.9,
                        "do_sample": True,
                        "return_full_text": False,
                        "stop": ["<|eot_id|>"],
                    },
                }
                return json.dumps(payload).encode("utf-8")

            def transform_output(self, output: bytes) -> str:
                # output is bytes — do not call .read()
                response = json.loads(output.decode("utf-8"))
                if isinstance(response, list):
                    return response[0].get("generated_text", "")
                return response.get("generated_text", str(response))

        llm = SagemakerEndpoint(
            endpoint_name=endpoint_name,
            region_name=region,
            content_handler=Llama32ContentHandler(),
            model_kwargs={"temperature": temperature, "max_new_tokens": max_tokens},
        )
        logger.info(f"SageMaker LLM initialised: endpoint={endpoint_name}")
        return llm

    except ImportError as e:
        raise ImportError(
            f"Missing dependency for SageMaker: {e}\n"
            "Install: pip install langchain-aws boto3"
        )
