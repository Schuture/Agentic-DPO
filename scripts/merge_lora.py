"""Merge LoRA adapter into base model for vLLM inference.

Saves the full multimodal checkpoint (visual + LM) so vLLM can load it
with the same model_type as the original base model.
"""
import argparse
import json
import shutil
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file, save_file
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base_path = Path(args.base_model)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Merge LoRA into the CausalLM (text-only) model
    print(f"Loading CausalLM from {args.base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float16, device_map="cpu",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    print(f"Loading LoRA adapter from {args.adapter}...")
    model = PeftModel.from_pretrained(model, args.adapter)

    print("Merging adapter...")
    model = model.merge_and_unload()

    # Step 2: Get merged LM weights
    merged_state = model.state_dict()
    print(f"Merged CausalLM: {len(merged_state)} keys")

    # Step 3: Load full base model weights (including visual encoder + mtp)
    base_safetensor_files = sorted(base_path.glob("model*.safetensors"))
    base_state = {}
    for sf in base_safetensor_files:
        base_state.update(load_file(str(sf)))
    print(f"Base model: {len(base_state)} keys")

    # Step 4: Overwrite LM keys with merged weights
    # CausalLM keys (model.X) need mapping to base keys which may have different prefixes
    # e.g., model.layers.0.X -> model.language_model.layers.0.X (Qwen3.5 multimodal)
    #        lm_head.weight -> model.language_model.lm_head.weight
    updated = 0
    skipped = 0
    for key, val in merged_state.items():
        # Try direct match first
        if key in base_state:
            base_state[key] = val
            updated += 1
        else:
            # Try adding model.language_model prefix (for Qwen3.5 ConditionalGeneration)
            # Build candidate keys for different model architectures
            # CausalLM key "model.language_model.layers.0.X" -> base "language_model.model.layers.0.X" (Gemma3)
            # CausalLM key "model.layers.0.X" -> base "model.language_model.layers.0.X" (Qwen3.5)
            stripped = key.removeprefix("model.")
            candidates = [
                f"model.language_model.{stripped}",           # Qwen3.5: model.X -> model.language_model.X
                f"model.language_model.{key}",                # Qwen3.5: lm_head -> model.language_model.lm_head
                f"language_model.model.{stripped}",            # Gemma3: model.language_model.X -> language_model.model.X
                key.replace("model.language_model.", "language_model.model."),  # Gemma3 direct remap
            ]
            matched = False
            for candidate in candidates:
                if candidate in base_state:
                    base_state[candidate] = val
                    updated += 1
                    matched = True
                    break
            if not matched:
                skipped += 1
                if skipped <= 3:
                    print(f"  WARNING: merged key not in base: {key}")
    if skipped > 3:
        print(f"  ... and {skipped - 3} more unmatched keys")
    print(f"Updated {updated}/{len(merged_state)} keys in base state ({skipped} unmatched)")

    # Step 5: Save as single safetensors file with base model config
    # Break shared tensors (e.g., tied embed_tokens/lm_head) to avoid safetensors error
    for key in list(base_state.keys()):
        base_state[key] = base_state[key].clone()
    print(f"Saving to {output_path}...")
    save_file(base_state, str(output_path / "model.safetensors"))

    # Copy config, tokenizer, and other files from base model
    for fname in ["config.json", "generation_config.json", "preprocessor_config.json",
                  "video_preprocessor_config.json",
                  "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                  "special_tokens_map.json", "vocab.json", "merges.txt"]:
        src = base_path / fname
        if src.exists():
            shutil.copy2(str(src), str(output_path / fname))

    # Update safetensors index if needed
    index_file = output_path / "model.safetensors.index.json"
    if index_file.exists():
        index_file.unlink()  # Remove old index since we saved as single file

    # Note: base model tokenizer files are already copied above (lines 93-99).
    # Do NOT call tokenizer.save_pretrained() here — it would overwrite the base
    # model's tokenizer_config.json with the training env's version (TokenizersBackend).
    print("Done.")


if __name__ == "__main__":
    main()
