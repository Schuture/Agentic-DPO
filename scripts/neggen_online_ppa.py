"""Per-round online-PPA negative generation, strict visit budget.

Each refresh round only needs negatives for the exact ``n_visits`` (pair, variant)
tuples the trainer is going to consume in the next ``steps_per_round``
optimizer steps (``n_visits = steps_per_round × effective_batch_size``).
Generating more than that is wasted compute, so this wrapper:

1. Samples ``n_visits`` canonical pair indices uniformly without replacement
   (deterministic per ``round_seed``) — these are the pairs the trainer will
   iterate over in the round.
2. For each sampled pair, draws one schema variant ``phi ~ Unif(Phi)``, also
   deterministically per ``(pair_id, round_seed)`` (paper Algorithm 1).
3. Renders the (state, expert action) under the chosen variant via the
   registered ``PPARenderer``, runs vLLM K-sample once against the current
   student, and writes:

   - ``--lookup_out``  (the negatives lookup JSON keyed by
     ``f"{pair_id}__{variant}"``  → ``negative_step_text``), and
   - ``--subset_pairs_out``  (the canonical pair *subset* the trainer should
     read for this round, so its ``len(dataset) = n_visits`` and the round's
     ``max_steps = n_visits / effective_batch_size`` covers each pair once).

Pairs whose K candidates are all same-tool-as-expert are dropped (no fallback)
per paper §4.4. The dropped pairs simply get no entry in the lookup; the
trainer's ``OnlinePPAStepDataset`` falls back to an SFT-only sample
(``weight=0`` on the DPO term) for those.

Usage:
    python scripts/neggen_online_ppa.py \\
        --canonical_pairs data/step_pairs_canonical.json \\
        --domain stb \\
        --sampler_model /path/to/r0_warmup_merged \\
        --lookup_out  runs/.../negatives_lookup_r1.json \\
        --subset_pairs_out runs/.../subset_pairs_r1.json \\
        --n_visits 1200 --round_seed 4201 \\
        --K 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

# Allow `python scripts/neggen_online_ppa.py` from the project root by adding
# the parent directory to sys.path so `src.*` and `scripts.*` resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ppa_render import get_renderer
from src.training.agentic_dpo_negative_gen import generate_negatives


def _stable_seed(*parts) -> int:
    payload = "|".join(str(x) for x in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical_pairs", required=True,
                    help="Canonical (un-PPA-expanded) step pairs JSON.")
    ap.add_argument("--domain", default="stb",
                    help="Registered PPA renderer name (default: stb).")
    ap.add_argument("--sampler_model", required=True,
                    help="Warmed-up student checkpoint used for negative sampling.")
    ap.add_argument("--lookup_out", required=True,
                    help="Output negatives lookup JSON.")
    ap.add_argument("--subset_pairs_out", required=True,
                    help="Output canonical-pair subset JSON the trainer should "
                         "read for this round.")
    ap.add_argument("--n_visits", type=int, required=True,
                    help="Number of (pair, variant) tuples to cover in the "
                         "next training round = steps_per_round × eff_batch_size.")
    ap.add_argument("--round_seed", type=int, default=0,
                    help="Per-round seed used to draw the pair subset and the "
                         "variant per pair. Set differently for each round.")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--enforce_eager", action="store_true")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = ap.parse_args()

    canonical = json.load(open(args.canonical_pairs))
    print(f"Loaded {len(canonical)} canonical pairs.")
    if args.n_visits > len(canonical):
        raise ValueError(
            f"n_visits={args.n_visits} > #canonical_pairs={len(canonical)}; "
            f"either lower n_visits or fall back to all-pairs neggen."
        )
    renderer = get_renderer(args.domain)
    print(f"Renderer: {args.domain}, variants={renderer.variants}  "
          f"round_seed={args.round_seed}  n_visits={args.n_visits}")

    # 1) Pick the canonical pair subset for this round (deterministic).
    pair_rng = random.Random(_stable_seed("pair_subset", args.round_seed))
    indices = list(range(len(canonical)))
    pair_rng.shuffle(indices)
    chosen_indices = sorted(indices[: args.n_visits])
    subset = [canonical[i] for i in chosen_indices]

    # 2) For each chosen pair, draw a variant deterministically.
    expanded: list[dict] = []
    key_for_index: list[tuple[str, str]] = []
    variant_draw_counts: Counter = Counter()
    for p in subset:
        cid = p["pair_id"]
        v_rng = random.Random(_stable_seed("variant", cid, args.round_seed))
        v = v_rng.choice(renderer.variants)
        variant_draw_counts[v.lstrip("_")] += 1
        rendered = renderer.build_variant(p, v)
        expanded.append(rendered)
        key_for_index.append((cid, v.lstrip("_")))

    print(f"Subset: {len(expanded)} (pair, variant) tuples to neggen.")
    print("Per-variant draw counts: " + "  ".join(
        f"{vname}={variant_draw_counts[vname]}" for vname in
        sorted(variant_draw_counts)
    ))

    # 3) Run vLLM K-sample once over exactly those n_visits prompts.
    expanded = generate_negatives(
        expanded,
        model_path=args.sampler_model,
        K=args.K,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # 4) Build outputs.
    lookup: dict[str, str] = {}
    per_variant_counts = Counter()
    per_variant_kept = Counter()
    for (cid, vname), pair in zip(key_for_index, expanded):
        per_variant_counts[vname] += 1
        neg = pair.get("negative_step_text", "").strip()
        if neg:
            lookup[f"{cid}__{vname}"] = neg
            per_variant_kept[vname] += 1

    print()
    print("=== Per-variant retention ===")
    for v in renderer.variants:
        vn = v.lstrip("_")
        kept = per_variant_kept[vn]
        total = per_variant_counts[vn]
        pct = 100.0 * kept / max(total, 1)
        print(f"  {vn:10s} {kept}/{total}  ({pct:.1f}% kept)")
    print(f"  total lookup entries: {len(lookup)}/{len(expanded)} "
          f"({100.0*len(lookup)/max(len(expanded),1):.1f}% kept)")

    Path(args.lookup_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(lookup, open(args.lookup_out, "w"))
    print(f"Wrote lookup        -> {args.lookup_out}")

    # The subset file the trainer will read this round. Pair ids are the
    # ORIGINAL canonical ids; the trainer's OnlinePPAStepDataset matches them
    # against the lookup keys.
    Path(args.subset_pairs_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(subset, open(args.subset_pairs_out, "w"))
    print(f"Wrote subset pairs  -> {args.subset_pairs_out}")


if __name__ == "__main__":
    main()
