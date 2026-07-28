"""
Agentic-DPO Negative Generation: Generate student-sampled negatives via vLLM.

For each expert state, sample K candidate next-steps from the student model,
then select the hard negative (highest log-prob non-expert candidate).

Usage:
    python -m src.training.agentic_dpo_negative_gen \
        --step_pairs data/scd/step_pairs_raw.json \
        --model_path /mnt/realccvl15/ychen646/llms/Qwen3.5-2B \
        --output data/scd/step_pairs_student_sampled.json \
        --K 3 --temperature 0.7
"""

import argparse
import json
import re
from pathlib import Path

from vllm import LLM, SamplingParams


def load_step_pairs(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def build_prompts(pairs: list[dict], tokenizer, max_prompt_tokens: int = 0) -> tuple[list[str], list[int]]:
    """Build prompts from state_messages for each step pair.

    Each prompt = chat_template(state_messages + start of assistant turn).
    The model will generate the next step (action).

    Returns (prompts, valid_indices) where valid_indices maps back to original pairs.
    Prompts exceeding max_prompt_tokens are skipped.
    """
    prompts = []
    valid_indices = []
    n_skipped = 0
    for i, pair in enumerate(pairs):
        messages = pair["state_messages"]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        # Skip overly long prompts
        if max_prompt_tokens > 0:
            token_len = len(tokenizer.encode(prompt, add_special_tokens=False))
            if token_len >= max_prompt_tokens:
                n_skipped += 1
                continue
        prompts.append(prompt)
        valid_indices.append(i)
    if n_skipped > 0:
        print(f"  Skipped {n_skipped} prompts exceeding {max_prompt_tokens} tokens")
    return prompts, valid_indices


def parse_action_from_generation(text: str) -> tuple[str, str]:
    """Parse tool name and full action text from generated text.

    Supports two formats:
    1. ReAct: Action: tool_name\nAction Input: {...}
    2. Bracket: [FuncName(arg1="val1", arg2="val2")]

    Returns (tool_name, action_text).
    """
    text = text.strip()

    # Try bracket format first: [FuncName(...)]
    bracket_match = re.search(r'\[([^\[\]]+)\]', text)
    if bracket_match:
        inner = bracket_match.group(1)
        # Extract function name (before first parenthesis)
        paren_idx = inner.find('(')
        if paren_idx != -1:
            tool_name = inner[:paren_idx].strip()
            action_text = '[' + inner + ']'
            return tool_name, action_text

    # Try τ-bench ReAct format: Action:\n{"name": ..., "arguments": ...}
    # Take the last "Action:" occurrence (Thought may contain the word in prose).
    tau_match = re.search(r"Action:\s*\n?\s*(\{.*?\})\s*(?:\n|$)", text, re.S)
    if tau_match:
        raw = tau_match.group(1).strip()
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "name" in obj:
                tool_name = str(obj["name"]).strip()
                # Canonical reproduction — matches build_tau_multi_schema render_action _base
                action_text = f"Action:\n{json.dumps(obj, ensure_ascii=False)}"
                return tool_name, action_text
        except json.JSONDecodeError:
            pass

    # Try Hermes <tool_call> block: matches _json/_combined variants in PPA.
    tc_match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S)
    if tc_match:
        raw = tc_match.group(1).strip()
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "name" in obj:
                tool_name = str(obj["name"]).strip()
                action_text = f"<tool_call>\n{json.dumps(obj, ensure_ascii=False)}\n</tool_call>"
                return tool_name, action_text
        except json.JSONDecodeError:
            pass

    # Legacy ReAct format: Action: tool\nAction Input: {json}
    action_match = re.search(r"Action:\s*(\S[^\n]*)\n", text)
    if action_match:
        tool_name = action_match.group(1).strip()
        ai_match = re.search(r"Action Input:\s*(.*?)(?:\n|$)", text, re.S)
        if ai_match:
            action_input = ai_match.group(1).strip()
            action_text = f"Action: {tool_name}\nAction Input: {action_input}"
        else:
            action_text = f"Action: {tool_name}"
        return tool_name, action_text

    return "", text.strip()


def generate_negatives(
    pairs: list[dict],
    model_path: str,
    K: int = 3,
    temperature: float = 0.7,
    max_tokens: int = 256,
    max_model_len: int = 4096,
    tensor_parallel_size: int = 1,
    dtype: str = "auto",
    enforce_eager: bool = False,
    gpu_memory_utilization: float = 0.85,
) -> list[dict]:
    """Generate student-sampled negatives for all step pairs.

    For each state, generates K candidates and selects the hard negative:
    the highest-probability candidate that differs from the expert action.
    """
    print(f"Loading model from {model_path}...")

    llm_kwargs = dict(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        dtype=dtype,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    if enforce_eager:
        llm_kwargs["enforce_eager"] = True

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    # Build prompts (skip those exceeding max context - reserve space for generation)
    print("Building prompts...")
    max_prompt_tokens = max_model_len - max_tokens if max_model_len > max_tokens else 0
    prompts, valid_indices = build_prompts(pairs, tokenizer, max_prompt_tokens=max_prompt_tokens)
    print(f"  Built {len(prompts)} prompts (from {len(pairs)} pairs)")

    # Generate K candidates per prompt
    # We generate K * len(prompts) total, by repeating each prompt K times
    all_prompts = []
    prompt_indices = []  # maps to valid_indices index
    for i, prompt in enumerate(prompts):
        for k in range(K):
            all_prompts.append(prompt)
            prompt_indices.append(valid_indices[i])

    print(f"Generating {len(all_prompts)} candidates ({K} per state)...")
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.95,
    )

    # Generate in chunks to avoid vLLM V1 engine crash with too many prompts
    CHUNK_SIZE = 10000
    outputs = []
    for chunk_start in range(0, len(all_prompts), CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(all_prompts))
        chunk = all_prompts[chunk_start:chunk_end]
        print(f"  Chunk {chunk_start//CHUNK_SIZE + 1}/{(len(all_prompts) + CHUNK_SIZE - 1)//CHUNK_SIZE}: {len(chunk)} prompts")
        # Recreate LLM for each chunk to avoid V1 multi-generate crash
        if chunk_start > 0:
            del llm
            import gc; gc.collect()
            import torch; torch.cuda.empty_cache()
            llm = LLM(**llm_kwargs)
        chunk_outputs = llm.generate(chunk, sampling_params)
        outputs.extend(chunk_outputs)

    # Group outputs by original pair index
    candidates_by_pair = {}
    for output, pair_idx in zip(outputs, prompt_indices):
        if pair_idx not in candidates_by_pair:
            candidates_by_pair[pair_idx] = []
        gen_text = output.outputs[0].text
        # Get cumulative log-prob
        logprobs = output.outputs[0].cumulative_logprob
        tool_name, action_text = parse_action_from_generation(gen_text)
        candidates_by_pair[pair_idx].append({
            "text": gen_text,
            "action_text": action_text,
            "tool_name": tool_name,
            "logprob": logprobs if logprobs is not None else 0.0,
        })

    # Select hard negatives.
    # Strict policy (matches paper §4.4): keep only candidates whose tool name
    # differs from the expert; pairs with no different-tool candidate are
    # discarded for this training pass (negative_step_text="" → trainer skip).
    # No fallback to same-tool / different-args candidates.
    print("Selecting hard negatives...")
    n_selected = 0
    n_discarded = 0

    for i, pair in enumerate(pairs):
        candidates = candidates_by_pair.get(i, [])
        expert_tool = pair["expert_tool_name"]

        valid_candidates = [
            c for c in candidates
            if c["tool_name"] != expert_tool and c["action_text"].strip()
        ]

        if valid_candidates:
            hard_neg = max(valid_candidates, key=lambda c: c["logprob"])
            pair["negative_step_text"] = hard_neg["action_text"]
            pair["negative_type"] = "student_sampled"
            n_selected += 1
        else:
            pair["negative_step_text"] = ""
            pair["negative_type"] = "discarded_no_diff_tool"
            n_discarded += 1

    discard_pct = 100.0 * n_discarded / max(len(pairs), 1)
    print(f"  Selected {n_selected} hard negatives, {n_discarded} discarded "
          f"(no candidate with different tool name) [{discard_pct:.1f}%]")
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate student-sampled negatives via vLLM")
    parser.add_argument("--step_pairs", type=str, required=True, help="Path to raw step pairs JSON")
    parser.add_argument("--model_path", type=str, required=True, help="Path to student model")
    parser.add_argument("--output", type=str, required=True, help="Output path for step pairs with negatives")
    parser.add_argument("--K", type=int, default=3, help="Number of candidates per state")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = parser.parse_args()

    # Load pairs
    pairs = load_step_pairs(args.step_pairs)
    print(f"Loaded {len(pairs)} step pairs")

    # Generate negatives
    pairs = generate_negatives(
        pairs,
        model_path=args.model_path,
        K=args.K,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(pairs)} pairs to {args.output}")


if __name__ == "__main__":
    main()
