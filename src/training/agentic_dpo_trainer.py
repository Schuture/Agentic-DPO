"""Agentic-DPO trainer.

Single objective:

    L = sft_weight · L_SFT(expert)
        + dpo_weight · L_DPO(expert vs. negative, length-scaled β_eff)

with `β_eff = β / max(|a⁺|, |a⁻|)^α`. Reference logprobs are passed in via the
ref_cache (paper recipe: zero-ref, Δ_ref ≡ 0). Every sample executes exactly
two forward passes (expert + negative), so the DDP graph is identical across
ranks; in R0 SFT warm-up (`dpo_weight = 0`) we take a fast path that skips
the negative forward pass.

Online PPA (paper §3.3): when ``online_ppa_domain`` is set, the trainer reads
canonical (un-rendered) step pairs and per gradient step samples a schema
view ``φ ∼ Unif(Φ)``, rendering ``(state, expert action, negative action)``
on the fly via the registered :class:`PPARenderer`. The ``negatives_lookup``
JSON is keyed by ``f"{pair_id}__{variant_name}"``.

Usage:
    python -m src.training.agentic_dpo_trainer \\
        --model_name qwen3.5-2b \\
        --step_pairs data/step_pairs_canonical.json \\
        --ref_cache  data/ref_logprobs_zero.json \\
        --online_ppa_domain stb \\
        --negatives_lookup runs/.../negatives_lookup_r1.json \\
        --beta 0.008 --alpha 0.5 \\
        --sft_weight 0.5 --dpo_weight 1.0 \\
        --epochs 1 --max_steps 75
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import TrainingArguments, Trainer
from torch.utils.data import Dataset as TorchDataset

from .sft_lora import SFTConfig, load_model_and_tokenizer, setup_lora


@dataclass
class AgenticDPOConfig(SFTConfig):
    """Agentic-DPO training configuration."""
    # Loss weights and length-scaled implicit reward margin.
    beta: float = 0.008
    alpha: float = 0.5
    sft_weight: float = 0.5
    dpo_weight: float = 1.0  # 0.0 during R0 SFT warm-up; 1.0 during refresh rounds.

    # Data.
    step_pairs_path: str = ""
    ref_cache_path: str = ""

    # Online PPA (paper §3.3).
    online_ppa_domain: str = ""             # "" | "stb" | …
    negatives_lookup_path: str = ""         # required for refresh rounds

    # Reproducibility.
    seed: int = 42


class AgenticDPODataCollator:
    """Pad expert + negative pairwise sequences and decision-token labels."""

    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, features: list[dict]) -> dict:
        expert_ids = self._pad([f["expert_input_ids"] for f in features])
        neg_ids = self._pad([f["negative_input_ids"] for f in features])
        return {
            "expert_input_ids": expert_ids,
            "expert_labels_sft": self._pad([f["expert_labels_sft"] for f in features], pad_val=-100),
            "expert_labels_dpo": self._pad([f["expert_labels_dpo"] for f in features], pad_val=-100),
            "expert_attention_mask": self._pad([f["expert_attention_mask"] for f in features]),
            "negative_input_ids": neg_ids,
            "negative_labels_dpo": self._pad([f["negative_labels_dpo"] for f in features], pad_val=-100),
            "negative_attention_mask": self._pad([f["negative_attention_mask"] for f in features]),
            "weights": torch.tensor([f["weight"] for f in features], dtype=torch.float32),
            "ref_logprob_expert": torch.tensor(
                [f["ref_logprob_expert"] for f in features], dtype=torch.float32),
            "ref_logprob_negative": torch.tensor(
                [f["ref_logprob_negative"] for f in features], dtype=torch.float32),
        }

    def _pad(self, sequences: list[list[int]], pad_val: int = None) -> torch.Tensor:
        if pad_val is None:
            pad_val = self.pad_id
        max_len = min(max(len(s) for s in sequences), self.max_length)
        padded = []
        for seq in sequences:
            seq = seq[:max_len]
            padded.append(seq + [pad_val] * (max_len - len(seq)))
        return torch.tensor(padded, dtype=torch.long)


class AgenticDPOTrainer(Trainer):
    """Length-scaled DPO with SFT anchor, two forward passes per sample."""

    def __init__(self, dpo_config: AgenticDPOConfig, **kwargs):
        super().__init__(**kwargs)
        self.dpo_config = dpo_config

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss = self._compute_dpo_loss(model, inputs)
        return (loss, {}) if return_outputs else loss

    def _compute_dpo_loss(self, model, inputs):
        cfg = self.dpo_config

        # Forward 1: expert (always).
        expert_logits = model(
            input_ids=inputs["expert_input_ids"],
            attention_mask=inputs["expert_attention_mask"],
        ).logits
        sft_loss = self._ce_loss(expert_logits, inputs["expert_labels_sft"])

        # SFT-warm-up fast path: skip the negative forward when its coefficient
        # is 0. DDP-safe because dpo_weight is a config scalar so all ranks
        # branch identically.
        if cfg.dpo_weight == 0.0:
            return cfg.sft_weight * sft_loss

        expert_logprobs = self._sum_logprobs(expert_logits, inputs["expert_labels_dpo"])

        # Forward 2: negative.
        neg_logits = model(
            input_ids=inputs["negative_input_ids"],
            attention_mask=inputs["negative_attention_mask"],
        ).logits
        neg_logprobs = self._sum_logprobs(neg_logits, inputs["negative_labels_dpo"])

        # Length-scaled implicit-reward margin: β_eff = β / max(|a⁺|,|a⁻|)^α.
        expert_lens = (inputs["expert_labels_dpo"][..., 1:] != -100).sum(dim=1).float()
        neg_lens = (inputs["negative_labels_dpo"][..., 1:] != -100).sum(dim=1).float()
        max_lens = torch.max(expert_lens, neg_lens).clamp(min=1)
        beta_eff = cfg.beta / (max_lens ** cfg.alpha) if cfg.alpha > 0 else cfg.beta

        ref_expert = inputs["ref_logprob_expert"].to(model.device)
        ref_neg = inputs["ref_logprob_negative"].to(model.device)
        margin = beta_eff * ((expert_logprobs - neg_logprobs) - (ref_expert - ref_neg))

        weights = inputs["weights"].to(model.device)
        dpo_loss = (-weights * F.logsigmoid(margin)).mean()

        return cfg.dpo_weight * dpo_loss + cfg.sft_weight * sft_loss

    @staticmethod
    def _sum_logprobs(logits, labels):
        """Sum of log-probs over labeled tokens. Returns (batch,)."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        mask = (shift_labels != -100).float()
        ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.clamp(min=0).view(-1),
            reduction='none',
        ).view(shift_labels.shape)
        return (-ce * mask).sum(dim=1)

    @staticmethod
    def _ce_loss(logits, labels):
        """Mean cross-entropy. Returns scalar."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )


# ── Decision-token masks ────────────────────────────────────────────────────

def _decision_mask_pair(
    expert_action_tokens: list[int],
    negative_action_tokens: list[int],
) -> tuple[list[int], list[int]]:
    """Compare expert and negative action tokens position-by-position.

    Returns ``(expert_mask, negative_mask)`` of the same length as the
    respective action token sequences:

        0 = format token (expert[t] == negative[t])
        1 = decision token (expert[t] != negative[t] or beyond the other length)

    Decision tokens carry the contrastive DPO gradient; format tokens are
    only supervised by the SFT term.
    """
    e_mask: list[int] = []
    for i, tok in enumerate(expert_action_tokens):
        if i < len(negative_action_tokens) and tok == negative_action_tokens[i]:
            e_mask.append(0)
        else:
            e_mask.append(1)
    n_mask: list[int] = []
    for i, tok in enumerate(negative_action_tokens):
        if i < len(expert_action_tokens) and tok == expert_action_tokens[i]:
            n_mask.append(0)
        else:
            n_mask.append(1)
    return e_mask, n_mask


# ── Tokenization ────────────────────────────────────────────────────────────

def _tokenize_state_step(
    tokenizer,
    state_text: str,
    step_text: str,
    max_length: int,
) -> Optional[dict]:
    state_tokens = tokenizer.encode(state_text, add_special_tokens=False)
    eos_token = tokenizer.eos_token or "<|im_end|>"
    full_text = state_text + step_text + eos_token
    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)

    if len(full_tokens) > max_length:
        full_tokens = full_tokens[:max_length]
        if len(full_tokens) <= len(state_tokens):
            return None

    n_state = min(len(state_tokens), len(full_tokens))
    return {
        "input_ids": full_tokens,
        "labels_sft": list(full_tokens),
        "attention_mask": [1] * len(full_tokens),
        "n_state": n_state,
        "action_tokens": full_tokens[n_state:],
    }


def _tokenize_one_pair(
    pair: dict,
    ref_cache: dict,
    tokenizer,
    max_length: int,
) -> Optional[dict]:
    """Tokenize a (state, expert, negative) triple into a sample dict.

    The expert action text (``expert_action_text``) is what the DPO term
    contrasts; the SFT label covers the full expert turn (state-prefix masked).

    Weight policy:
      - pair with no negative text → caller already filled a copy-of-expert
        dummy and is expected to set weight=0 outside this function.
      - identical expert/negative tokens (decision_ratio == 0) → weight=0
        (pure SFT, DPO term is degenerate).
      - otherwise → weight=1 (uniform across pairs; paper recipe).
    """
    if not pair.get("negative_step_text"):
        return None

    state_text = tokenizer.apply_chat_template(
        pair["state_messages"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    expert_text = pair["expert_action_text"]
    negative_text = pair["negative_step_text"]

    expert_result = _tokenize_state_step(tokenizer, state_text, expert_text, max_length)
    if expert_result is None:
        return None
    neg_result = _tokenize_state_step(tokenizer, state_text, negative_text, max_length)
    if neg_result is None:
        return None

    expert_action = expert_result["action_tokens"]
    neg_action = neg_result["action_tokens"]
    e_dmask, n_dmask = _decision_mask_pair(expert_action, neg_action)

    n_state_e = expert_result["n_state"]
    expert_labels_dpo = [-100] * n_state_e + list(expert_action)
    n_state_n = neg_result["n_state"]
    neg_labels_dpo = [-100] * n_state_n + list(neg_action)

    n_decision = sum(e_dmask)
    weight = 0.0 if n_decision == 0 else 1.0  # uniform per paper recipe

    pair_id = pair["pair_id"]
    ref_expert = ref_cache.get(pair_id, {}).get("expert", 0.0)
    ref_neg = ref_cache.get(pair_id, {}).get("negative", 0.0)

    return {
        "expert_input_ids": expert_result["input_ids"],
        "expert_labels_sft": expert_result["labels_sft"],
        "expert_labels_dpo": expert_labels_dpo,
        "expert_attention_mask": expert_result["attention_mask"],
        "negative_input_ids": neg_result["input_ids"],
        "negative_labels_dpo": neg_labels_dpo,
        "negative_attention_mask": neg_result["attention_mask"],
        "ref_logprob_expert": ref_expert,
        "ref_logprob_negative": ref_neg,
        "weight": weight,
    }


# ── Online-PPA dataset ──────────────────────────────────────────────────────

class OnlinePPAStepDataset(TorchDataset):
    """On-the-fly Policy-Preserving Augmentation.

    Stores canonical (un-rendered) step pairs and a precomputed lookup of
    one-step student negatives keyed by ``f"{canonical_pair_id}__{variant}"``.
    Each ``__getitem__`` samples a variant ``φ ∼ Unif(Φ)``, renders the pair
    under ``φ`` via the per-domain :class:`PPARenderer`, and tokenizes the
    resulting (state, expert, negative) triple on the fly.

    DDP-safety: every sample yields two forward passes (expert + negative)
    regardless of whether a real negative was available. When no variant has
    a precomputed negative, the sample falls back to a copy-of-expert dummy
    with ``weight = 0`` so the DPO term zeroes out cleanly while the
    per-step graph stays identical across ranks.
    """

    _SEPARATOR = "__"

    def __init__(
        self,
        canonical_pairs: list[dict],
        renderer,
        tokenizer,
        max_length: int,
        ref_cache: Optional[dict] = None,
        negatives_lookup: Optional[dict] = None,
        rng_seed: int = 0,
    ):
        self.canonical_pairs = canonical_pairs
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ref_cache = ref_cache or {}
        self.negatives_lookup = negatives_lookup or {}
        self._rng = random.Random(rng_seed)
        self._sft_only = not negatives_lookup

    def __len__(self):
        return len(self.canonical_pairs)

    def _key(self, pair_id: str, variant: str) -> str:
        return f"{pair_id}{self._SEPARATOR}{variant.lstrip('_')}"

    def __getitem__(self, idx: int) -> dict:
        canonical = self.canonical_pairs[idx]
        canonical_id = canonical["pair_id"]
        variants = list(self.renderer.variants)
        self._rng.shuffle(variants)

        chosen_variant = None
        chosen_neg_text = ""
        if not self._sft_only:
            for v in variants:
                neg_text = self.negatives_lookup.get(self._key(canonical_id, v))
                if neg_text:
                    chosen_variant = v
                    chosen_neg_text = neg_text
                    break

        if chosen_variant is None:
            # SFT-only fallback: render any variant, fill negative with a copy
            # of expert so the second forward stays well-formed; weight = 0
            # zeroes the DPO term.
            chosen_variant = variants[0]
            rendered = self.renderer.build_variant(canonical, chosen_variant)
            rendered["negative_step_text"] = rendered["expert_action_text"]
            sample = _tokenize_one_pair(rendered, self.ref_cache, self.tokenizer, self.max_length)
            if sample is None:
                return self.__getitem__((idx + 1) % len(self))
            sample["weight"] = 0.0
            sample["ref_logprob_expert"] = 0.0
            sample["ref_logprob_negative"] = 0.0
            return sample

        rendered = self.renderer.build_variant(canonical, chosen_variant)
        rendered["negative_step_text"] = chosen_neg_text
        ref_key = self._key(canonical_id, chosen_variant)
        rendered["pair_id"] = ref_key  # keys the ref cache lookup
        local_ref = {ref_key: self.ref_cache.get(ref_key, {"expert": 0.0, "negative": 0.0})}
        sample = _tokenize_one_pair(rendered, local_ref, self.tokenizer, self.max_length)
        if sample is None:
            return self.__getitem__((idx + 1) % len(self))
        return sample


# ── Training entry point ───────────────────────────────────────────────────

def train_agentic_dpo(config: AgenticDPOConfig):
    print("=== Agentic-DPO Training ===")
    print(f"Model:        {config.model_name}")
    print(f"β={config.beta}  α={config.alpha}  sft_weight={config.sft_weight}  "
          f"dpo_weight={config.dpo_weight}")

    if not config.output_dir:
        raise ValueError("output_dir must be set")
    if not config.online_ppa_domain:
        raise ValueError(
            "online_ppa_domain must be set (paper recipe uses online PPA only); "
            "see scripts/build_stb_multi_schema.py for the STB renderer."
        )

    model, tokenizer = load_model_and_tokenizer(config)
    model = setup_lora(model, config)

    print(f"Loading step pairs from {config.step_pairs_path}...")
    with open(config.step_pairs_path) as f:
        pairs = json.load(f)
    print(f"  Loaded {len(pairs)} pairs")

    # Reference logprobs: paper recipe uses the zero-ref file produced by the
    # shell pipeline (Δ_ref ≡ 0). Loaded as-is; missing keys default to 0.0
    # inside :class:`OnlinePPAStepDataset`.
    ref_cache: dict = {}
    if config.ref_cache_path and os.path.exists(config.ref_cache_path):
        print(f"Loading reference cache from {config.ref_cache_path}...")
        with open(config.ref_cache_path) as f:
            ref_cache = json.load(f)

    # Online PPA dataset.
    from ..data.ppa_render import get_renderer
    renderer = get_renderer(config.online_ppa_domain)
    print(f"Online PPA: domain={config.online_ppa_domain}, variants={renderer.variants}")
    negatives_lookup: dict = {}
    if config.negatives_lookup_path and os.path.exists(config.negatives_lookup_path):
        with open(config.negatives_lookup_path) as f:
            negatives_lookup = json.load(f)
        print(f"  Loaded negatives lookup: {len(negatives_lookup)} (pair_id, variant) entries")
    else:
        print(f"  No negatives_lookup → SFT-only warmup mode (weight=0 on every DPO term)")
    dataset = OnlinePPAStepDataset(
        canonical_pairs=pairs,
        renderer=renderer,
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        ref_cache=ref_cache,
        negatives_lookup=negatives_lookup,
        rng_seed=config.seed,
    )
    print(f"Total training samples: {len(dataset)}")

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        max_steps=config.max_steps if config.max_steps > 0 else -1,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        report_to="none",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        seed=config.seed,
    )

    collator = AgenticDPODataCollator(tokenizer, max_length=config.max_seq_length)
    trainer = AgenticDPOTrainer(
        dpo_config=config,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"Model saved to {config.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Agentic-DPO training")
    # Model + data
    parser.add_argument("--model_name", type=str, default="qwen3.5-2b")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--step_pairs", type=str, required=True)
    parser.add_argument("--ref_cache", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)

    # Loss
    parser.add_argument("--beta", type=float, default=0.008)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--sft_weight", type=float, default=0.5)
    parser.add_argument("--dpo_weight", type=float, default=1.0,
                        help="Set to 0.0 for R0 SFT warm-up, 1.0 during refresh rounds.")

    # Online PPA (required)
    parser.add_argument("--online_ppa_domain", type=str, required=True,
                        choices=["stb"],
                        help="Registered PPA renderer name. Required.")
    parser.add_argument("--negatives_lookup", type=str, default="",
                        help="Path to negatives lookup JSON keyed by '{pair_id}__{variant}'. "
                             "Omit for SFT-only warm-up.")

    # Training
    parser.add_argument("--quantization", type=str, default="int4",
                        choices=["", "int4", "int8"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="If >0, cap training at this many optimizer steps.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = AgenticDPOConfig(
        model_name=args.model_name,
        model_path=args.model_path,
        step_pairs_path=args.step_pairs,
        ref_cache_path=args.ref_cache,
        output_dir=args.output_dir,
        beta=args.beta,
        alpha=args.alpha,
        sft_weight=args.sft_weight,
        dpo_weight=args.dpo_weight,
        online_ppa_domain=args.online_ppa_domain,
        negatives_lookup_path=args.negatives_lookup,
        quantization=args.quantization,
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        seed=args.seed,
    )
    train_agentic_dpo(config)


if __name__ == "__main__":
    main()
