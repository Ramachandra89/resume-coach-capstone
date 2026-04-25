"""
Resume Coach - LLM Fine-Tuning Code
=====================================
Fine-tunes meta-llama/Llama-3.1-8B-Instruct on a resume coaching dataset
using QLoRA (Quantized Low-Rank Adaptation) for parameter-efficient training.

Why QLoRA:
  - Full fine-tuning of 7B model requires ~112GB VRAM (8x A100)
  - QLoRA + 4-bit quantization reduces this to ~10GB VRAM
  - Achieves comparable performance at fraction of the cost
  - Trainable on a single A10G (ml.g5.2xlarge) or A100 (ml.p4d instance)

Dataset format expected:
  training_data.jsonl — each line is a JSON object:
  {
    "instruction": "system prompt + task description",
    "input": "resume text + job description",
    "output": "expected coaching report or response"
  }

Usage:
  # Prepare dataset
  python fine_tune.py --mode prepare --data_dir ./data/raw

  # Train (requires GPU)
  python fine_tune.py --mode train --data_path ./data/prepared/train.jsonl

  # Evaluate
  python fine_tune.py --mode evaluate --model_path ./output/checkpoint-final

  # Push to S3
  python fine_tune.py --mode push --model_path ./output/checkpoint-final

SageMaker Training Job:
  See finetune_sagemaker_job.py for running as a managed SageMaker training job.
"""

import json
import logging
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

@dataclass
class FineTuningConfig:
    # Model
    base_model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir: str = "./output/resume-coach-ft"
    
    # QLoRA hyperparameters
    # Rationale:
    #   r=16: rank of LoRA matrices. Higher = more capacity but more params.
    #         16 is standard for instruction-following tasks.
    #   alpha=32: scaling factor. Rule of thumb: alpha = 2*r
    #   dropout=0.05: small dropout prevents overfitting on small datasets
    #   target_modules: attention projection layers only (not MLP)
    #         These are the standard Llama target modules for LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Training hyperparameters
    # Rationale:
    #   num_epochs=3: standard for instruction-following fine-tuning
    #                 More epochs → overfitting on small datasets
    #   batch_size=4: fits in g4dn.xlarge VRAM with gradient accumulation
    #   gradient_accumulation=4: effective batch = 4*4 = 16
    #                            Larger effective batch stabilizes training
    #   lr=2e-4: standard for QLoRA; larger than full fine-tuning
    #            because LoRA parameters start from 0
    #   warmup=100: ~10% of total steps; prevents lr spike at start
    #   max_seq_len=4096: covers most resume+JD+response combinations
    num_epochs: int = 3
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    lr_scheduler: str = "cosine"
    max_seq_length: int = 4096
    weight_decay: float = 0.001
    
    # Logging
    logging_steps: int = 25
    eval_steps: int = 100
    save_steps: int = 200
    
    # Data
    train_data_path: str = "./data/prepared/train.jsonl"
    eval_data_path: str = "./data/prepared/eval.jsonl"
    
    # Quantization
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True


# ─────────────────────────────────────────────
# DATASET PREPARATION
# ─────────────────────────────────────────────

class ResumeCoachDatasetPreparator:
    """
    Prepares training data for fine-tuning.
    
    Input: Raw {resume, job_description, coaching_report} pairs
    Output: Formatted JSONL with instruction/input/output structure
    
    We use the Llama 3.1 chat format with special header tokens.
    """
    
    SYSTEM_PROMPT = """You are an elite career coach and ATS optimization expert. 
Analyze the provided resume and job description, then generate a comprehensive 
coaching report with fit assessment, gap analysis, strengths, and actionable recommendations.
Always respond with valid JSON following the specified schema."""

    def format_training_example(
        self,
        resume_text: str,
        job_description: str,
        coaching_report: Dict[str, Any],
        example_type: str = "full_report"
    ) -> Dict[str, str]:
        """Format a single training example in Llama 3.1 chat format."""

        if example_type == "full_report":
            instruction = (
                "Analyze the candidate's resume against the job description and provide "
                "a structured coaching report in JSON format."
            )
            input_text = f"RESUME:\n{resume_text}\n\nJOB DESCRIPTION:\n{job_description}"
            output = json.dumps(coaching_report, indent=2)

        elif example_type == "gap_analysis":
            instruction = (
                "Identify the key gaps between the candidate's background and the job requirements."
            )
            input_text = f"RESUME:\n{resume_text}\n\nJOB DESCRIPTION:\n{job_description}"
            output = json.dumps({
                "gaps": coaching_report.get("gaps", []),
                "missing_keywords": coaching_report.get("missing_keywords", [])
            }, indent=2)

        # Llama 3.1 chat format with special header tokens
        formatted_text = (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n{self.SYSTEM_PROMPT}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{instruction}\n\n{input_text}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>"
        )

        return {
            "text": formatted_text,
            "instruction": instruction,
            "input": input_text,
            "output": output,
        }

    def prepare_dataset(
        self,
        raw_data_path: str,
        output_dir: str,
        train_split: float = 0.9,
    ) -> Dict[str, str]:
        """
        Prepare raw data into train/eval splits.
        
        Expected raw data format (raw_data.jsonl):
        {"resume": "...", "job_description": "...", "coaching_report": {...}}
        
        Returns paths to train and eval files.
        """
        import random
        
        raw_path = Path(raw_data_path)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not raw_path.exists():
            logger.warning(f"Raw data not found at {raw_path}. Generating synthetic examples.")
            examples = self._generate_synthetic_examples(n=100)
        else:
            examples = []
            with open(raw_path) as f:
                for line in f:
                    raw = json.loads(line.strip())
                    formatted = self.format_training_example(
                        resume_text=raw["resume"],
                        job_description=raw["job_description"],
                        coaching_report=raw["coaching_report"],
                    )
                    examples.append(formatted)
        
        # Shuffle and split
        random.shuffle(examples)
        split_idx = int(len(examples) * train_split)
        train_examples = examples[:split_idx]
        eval_examples = examples[split_idx:]
        
        # Write JSONL files
        train_path = output_path / "train.jsonl"
        eval_path = output_path / "eval.jsonl"
        
        with open(train_path, "w") as f:
            for ex in train_examples:
                f.write(json.dumps(ex) + "\n")
        
        with open(eval_path, "w") as f:
            for ex in eval_examples:
                f.write(json.dumps(ex) + "\n")
        
        logger.info(f"Dataset prepared: {len(train_examples)} train, {len(eval_examples)} eval")
        logger.info(f"Train: {train_path}")
        logger.info(f"Eval: {eval_path}")
        
        return {"train": str(train_path), "eval": str(eval_path)}

    def _generate_synthetic_examples(self, n: int = 100) -> List[Dict]:
        """
        Generate synthetic training examples for demonstration.
        In production, replace with real resume/JD/report pairs.
        These synthetic examples show the data structure needed.
        """
        templates = [
            {
                "resume": "John Smith | john@email.com\n\nSoftware Engineer with 4 years Python experience.\nSkills: Python, Django, PostgreSQL, Docker\nExperience: Backend Developer at TechCorp (2020-2024)\n- Built REST APIs serving 100k daily users\n- Reduced query time by 40% via indexing\nEducation: BS Computer Science, State University 2020",
                "job_description": "Senior Python Engineer\nRequirements: 5+ years Python, AWS experience required, Kubernetes preferred\nResponsibilities: Design scalable microservices, mentor junior developers",
                "coaching_report": {
                    "overall_fit_score": 68,
                    "ats_score": 72,
                    "fit_verdict": "Good Match",
                    "gaps": [
                        {"title": "AWS Experience", "severity": "Critical", "detail": "Role requires AWS but resume lacks cloud experience", "mitigation": "Add any AWS projects or get AWS Cloud Practitioner cert"},
                        {"title": "Years of Experience", "severity": "Moderate", "detail": "4 years vs 5 years required", "mitigation": "Emphasize impact over tenure"}
                    ],
                    "strengths": [
                        {"title": "Strong Python Background", "detail": "4 years of core Python directly relevant", "how_to_emphasize": "Lead with Python impact metrics"},
                    ],
                    "missing_keywords": ["AWS", "Kubernetes", "microservices", "senior"],
                    "present_keywords": ["Python", "Django", "PostgreSQL", "Docker", "REST API"],
                    "application_recommendation": "Recommend Applying",
                    "fit_summary": "Strong Python match with cloud experience gap. Apply and address AWS gap in cover letter."
                }
            }
        ]
        
        # Replicate with variations to reach n examples
        examples = []
        for i in range(n):
            template = templates[i % len(templates)]
            formatted = self.format_training_example(
                resume_text=template["resume"],
                job_description=template["job_description"],
                coaching_report=template["coaching_report"],
            )
            examples.append(formatted)
        
        return examples


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train(config: FineTuningConfig):
    """
    Fine-tune Llama-3.1-8B-Instruct with QLoRA.
    Requires: transformers, peft, trl, bitsandbytes, accelerate
    """
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
        from trl import SFTTrainer
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}\n"
            "Install: pip install transformers peft trl bitsandbytes accelerate datasets"
        )

    logger.info(f"Starting QLoRA fine-tuning of {config.base_model_id}")
    logger.info(f"Config: r={config.lora_r}, alpha={config.lora_alpha}, lr={config.learning_rate}")
    logger.info(f"Training for {config.num_epochs} epochs with batch_size={config.per_device_batch_size}")

    # ── Load tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_id,
        trust_remote_code=True,
        padding_side="right",  # Required for SFT with padding
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Quantization config ──
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.use_4bit,
        bnb_4bit_use_double_quant=config.use_nested_quant,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
    )

    # ── Load base model ──
    logger.info("Loading base model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # ── LoRA configuration ──
    # Rationale for target_modules:
    #   - q_proj, k_proj, v_proj: Query/Key/Value attention matrices
    #     These capture the model's attention patterns — most important for task adaptation
    #   - o_proj: Output projection of attention
    #   - gate_proj, up_proj, down_proj: MLP layers for knowledge injection
    #   Targeting all 7 modules maximizes adaptation capacity while keeping
    #   trainable params at ~0.5% of total (vs 100% for full fine-tuning)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ── Load dataset ──
    data_files = {
        "train": config.train_data_path,
        "validation": config.eval_data_path,
    }
    dataset = load_dataset("json", data_files=data_files)
    logger.info(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['validation'])} eval")

    # ── Training arguments ──
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        optim="paged_adamw_32bit",   # Memory-efficient optimizer for QLoRA
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=False,
        bf16=True,   # bfloat16 for A10G/A100 — better precision than fp16
        dataloader_num_workers=4,
        group_by_length=True,  # Pack similar-length sequences → fewer padding tokens
        report_to="none",  # Disable wandb/tensorboard unless configured
        run_name="resume-coach-qlora",
    )

    # ── Trainer ──
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        packing=True,  # Pack multiple short examples into one sequence
    )

    # ── Train ──
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete!")

    # ── Save ──
    final_model_path = os.path.join(config.output_dir, "checkpoint-final")
    trainer.model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"Model saved to {final_model_path}")

    return final_model_path


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_model(model_path: str, eval_data_path: str, n_samples: int = 20):
    """
    Evaluate fine-tuned model on held-out examples.
    Computes:
    - JSON validity rate (are outputs valid JSON?)
    - Schema completeness (are required keys present?)
    - Response coherence (manual spot-check)
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        from peft import PeftModel
        import torch
    except ImportError:
        raise ImportError("Install transformers, peft: pip install transformers peft")

    logger.info(f"Evaluating model at {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024)

    metrics = {"json_valid": 0, "schema_complete": 0, "total": 0}
    required_keys = ["overall_fit_score", "ats_score", "gaps", "strengths", "missing_keywords"]

    with open(eval_data_path) as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            example = json.loads(line.strip())
            # Strip assistant response — keep everything up to the assistant header
            assistant_header = "<|start_header_id|>assistant<|end_header_id|>\n\n"
            prompt = example["text"].split(assistant_header)[0] + assistant_header

            output = pipe(prompt, return_full_text=False)[0]["generated_text"]


            metrics["total"] += 1

            # Check JSON validity
            try:
                parsed = json.loads(output.strip())
                metrics["json_valid"] += 1
                # Check schema completeness
                if all(k in parsed for k in required_keys):
                    metrics["schema_complete"] += 1
            except json.JSONDecodeError:
                pass

    logger.info(f"\nEvaluation Results ({metrics['total']} samples):")
    logger.info(f"  JSON Valid Rate:       {metrics['json_valid']/metrics['total']*100:.1f}%")
    logger.info(f"  Schema Complete Rate:  {metrics['schema_complete']/metrics['total']*100:.1f}%")
    return metrics


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune resume coach LLM")
    parser.add_argument("--mode", choices=["prepare", "train", "evaluate"], required=True)
    parser.add_argument("--data_dir", default="./data/raw")
    parser.add_argument("--data_path", default="./data/prepared/train.jsonl")
    parser.add_argument("--model_path", default="./output/checkpoint-final")
    parser.add_argument("--output_dir", default="./output/resume-coach-ft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    args = parser.parse_args()

    if args.mode == "prepare":
        preparator = ResumeCoachDatasetPreparator()
        preparator.prepare_dataset(
            raw_data_path=os.path.join(args.data_dir, "raw_data.jsonl"),
            output_dir="./data/prepared",
        )
    elif args.mode == "train":
        config = FineTuningConfig(
            output_dir=args.output_dir,
            train_data_path=args.data_path,
            eval_data_path=args.data_path.replace("train", "eval"),
            num_epochs=args.epochs,
            learning_rate=args.lr,
            lora_r=args.lora_r,
            lora_alpha=args.lora_r * 2,
        )
        train(config)
    elif args.mode == "evaluate":
        evaluate_model(args.model_path, args.data_path.replace("train", "eval"))
