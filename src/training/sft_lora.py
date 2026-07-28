"""Model loader + LoRA wrapping helpers used by the Agentic-DPO trainer.

Only the pieces ``agentic_dpo_trainer.py`` consumes are kept here:
  - :class:`SFTConfig`               (the base config Agentic-DPO extends)
  - :func:`resolve_model_path`       (text-name → local path mapping)
  - :func:`load_model_and_tokenizer` (HF model + tokenizer, optional 4-bit quant)
  - :func:`setup_lora`               (peft LoRA wrapper)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


@dataclass
class SFTConfig:
    """Base training configuration shared with :class:`AgenticDPOConfig`."""
    # Model
    model_name: str = "qwen3.5-2b"
    model_path: str = ""        # auto-resolved if empty
    quantization: str = ""      # "", "int8", "int4"

    # LoRA
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training
    learning_rate: float = 2e-4
    num_epochs: int = 1
    max_steps: int = -1         # -1 → use num_epochs; >0 overrides for refresh rounds
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048
    bf16: bool = True
    gradient_checkpointing: bool = True
    lr_scheduler_type: str = "cosine"

    # Output
    output_dir: str = ""
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3


# Model name -> HuggingFace ID mapping.
MODEL_REGISTRY = {
    "qwen3.5-2b": "Qwen/Qwen3.5-2B",
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "qwen3.5-27b": "Qwen/Qwen3.5-27B",
}

LOCAL_MODEL_DIR = ""


def resolve_model_path(model_name: str) -> str:
    """Resolve a registered model name to a local path or HuggingFace ID."""
    candidates = [
        os.path.join(LOCAL_MODEL_DIR, model_name),
        os.path.join(LOCAL_MODEL_DIR, MODEL_REGISTRY.get(model_name, "")),
        os.path.join(LOCAL_MODEL_DIR, MODEL_REGISTRY.get(model_name, "").split("/")[-1]),
    ]
    for p in candidates:
        if p and os.path.isdir(p):
            return p
    return MODEL_REGISTRY.get(model_name, model_name)


def load_model_and_tokenizer(config: SFTConfig):
    """Load the base model and tokenizer with optional 4-/8-bit quantization."""
    model_path = config.model_path or resolve_model_path(config.model_name)
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # DDP: each rank loads the model on its own GPU.
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    device_map = {"": local_rank} if local_rank >= 0 else "auto"

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if config.bf16 else torch.float16,
        "device_map": device_map,
    }
    if config.quantization == "int4":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif config.quantization == "int8":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    if config.gradient_checkpointing:
        model.enable_input_require_grads()

    return model, tokenizer


def setup_lora(model, config: SFTConfig):
    """Wrap ``model`` with a LoRA adapter as specified by ``config``."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model
