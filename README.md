<h1 align="center">Agentic-DPO</h1>

<p align="center">
  <b>From Imitation to Agentic Policy Optimization on Expert Trajectories</b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2607.10601"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2607.10601-b31b1b.svg"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  <img alt="Python 3.10" src="https://img.shields.io/badge/python-3.10-blue.svg">
</p>

Supervised fine-tuning on expert agent trajectories teaches a model to *reproduce* the
expert's action sequence. It never teaches the model to *choose* the expert action over the
plausible mistake it would otherwise have made.

**Agentic-DPO** turns expert trajectories into state-conditioned preference supervision. At
each expert state we sample one-step actions from the *current* student, take its
highest-log-prob wrong action as the negative, and contrast it against the expert action with
a DPO-style objective. There are no environment rollouts and no reward model — it is a
drop-in replacement for SFT that costs about 1.6× an SFT step.

On τ-bench retail with a 9B backbone this recovers online GRPO's performance at
**8.5× lower per-step training cost** (1.6× vs. 13.6× SFT).

---

## Results

StableToolBench accuracy, **Qwen3.5-2B** — the configuration released here. Mean ± std over
three seeds. `Canon` is the six original STB subsets; `Pert` is the mean over nine
perturbation operators × six subsets at difficulty level 2.

| Method | Canon | Pert | Gap |
|---|---|---|---|
| Base (no training) | 45.2 ± 2.1 | 42.1 ± 2.3 | 3.1 |
| SFT | 57.1 ± 3.2 | 52.3 ± 3.0 | 4.8 |
| PPA + SFT | 83.4 ± 2.5 | 81.1 ± 2.8 | 2.3 |
| DFT | 85.2 ± 6.5 | 79.6 ± 6.6 | 5.6 |
| ETO (trajectory-level DPO) | 85.3 ± 2.2 | 81.3 ± 2.7 | 4.0 |
| GRPO (online RL) | 85.0 ± 2.9 | 79.8 ± 4.2 | 5.2 |
| **Agentic-DPO** | **90.9 ± 1.3** | **85.5 ± 1.5** | 5.4 |

At 9B the same recipe reaches **94.1 / 92.0** vs. SFT's 78.5 / 74.2. Full results across
StableToolBench, τ-bench retail, Mind2Web and zero-shot BFCL-v3 — plus the ablations, the
held-out perturbation study and the R-round analysis — are in the
[paper](https://arxiv.org/abs/2607.10601).

Three findings worth knowing before you run anything:

- **The SFT warm-up is not optional.** Without it the model collapses to 0.0 (every generation
  is rejected by the action parser). See Table 2 of the paper.
- **The SFT anchor is load-bearing.** Dropping `λ·L_SFT` costs ~18 points.
- **PPA is what closes the robustness gap.** Without it the canonical→perturbed gap widens
  from 5.4 to 11.4 points.

---

## Scope of this release

> [!IMPORTANT]
> This repository contains the **StableToolBench training pipeline on a Qwen3.5-2B backbone**.
> That is the configuration used for the paper's main 2B row and most of its ablations.

| | Status |
|---|---|
| Agentic-DPO trainer (loss, collator, decision-token masking) | ✅ included |
| Online-PPA renderer + negative generation (vLLM) | ✅ included |
| End-to-end STB pipeline (R0 warm-up → R1–R5 refresh rounds) | ✅ included |
| Canonical STB step pairs | ✅ included (`data/`) |
| **STB evaluation harness + perturbation generator** | ⏳ not yet released |
| **τ-bench / Mind2Web / BFCL-v3 pipelines** | ⏳ not yet released |
| Trained checkpoints | ⏳ not released |

The trainer is environment-agnostic; adding a new environment means registering one renderer
(see [Adding a new environment](#adding-a-new-environment)). Issues and PRs welcome.

---

## Installation

```bash
git clone https://github.com/Schuture/Agentic-DPO.git
cd Agentic-DPO
pip install -r requirements.txt
```

Qwen3.5 requires `transformers>=5.4.0`.

Training and negative generation can share one environment, but vLLM and the training stack
have conflicting pins often enough that we run them separately on our cluster:

```bash
# training env
python -m venv .venv-train && .venv-train/bin/pip install \
  torch transformers==5.4.0 peft datasets bitsandbytes safetensors accelerate

# inference env (negative generation only)
python -m venv .venv-vllm  && .venv-vllm/bin/pip install vllm
```

All commands below assume you run from the repository root — `src/data/ppa_render.py`
imports the renderer as `scripts.build_stb_multi_schema`, so the repo root must be on
`sys.path`.

---

## Quickstart

Reproduce the 2B StableToolBench run end to end:

```bash
export BASE_MODEL=/path/to/Qwen3.5-2B      # required
export SEED=42
export NUM_GPUS=4
export PYTHON_TRAIN=.venv-train/bin/python
export PYTHON_EVAL=.venv-vllm/bin/python
export TORCHRUN=.venv-train/bin/torchrun
export OUTPUT_DIR=./runs/stb_agentic_dpo_s${SEED}

bash scripts/run_stb_agentic_dpo.sh
```

**Runtime:** ~4 h per seed on 4× A6000 — 0.2 h warm-up, then 5 × (0.2 h negative generation
+ 0.5 h DPO). Peak disk under `OUTPUT_DIR` is a few hundred GB before the intermediate LoRA
directories are pruned.

**Output:** six merged checkpoints, `r0_warmup_merged` through `r5_dpo_merged`. Each is a
standalone model directory you can serve with vLLM. Select a round on a held-out validation
slice rather than defaulting to the last one — the paper's Figure 3 finds the canonical
optimum at R = 3 for 2B, R = 5 for 4B and R = 1 for 9B.

---

## Method

Two components on top of a standard DPO loss.

**1. State-conditioned negatives.** For every expert state, sample `K = 4` one-step
continuations from the current student. Discard any candidate that calls the same tool as the
expert; among the rest take the highest-log-prob one as the negative. If all `K` collapse to
the expert's tool, the pair is skipped for that pass. The negative therefore tracks the
student's *current* failure mode rather than a fixed adversary, which is why negatives are
refreshed every round.

**2. Policy-Preserving Augmentation (PPA).** At each gradient step, sample a schema view
`φ ~ Unif(Φ)` and render `(state, expert_action)` under it on the fly. For STB,
`Φ = {_base, _json, _rename, _combined}` — base ReAct, JSON-bracket action format, a
deterministic per-pair tool-name synonym swap, and both. The optimal policy is invariant to
these rewrites, so PPA adds robustness without changing what the model should do.

The objective is

```
L = λ · L_SFT(expert) + L_DPO(expert vs. negative)
```

with a length-scaled implicit reward margin `β_eff = β / max(|a⁺|, |a⁻|)^α` and a zero
reference (`Δ_ref ≡ 0`, so no reference-model forward pass). Every sample runs exactly two
forwards — expert and negative — so the DDP graph is identical across ranks. In R0,
`dpo_weight == 0` takes a fast path that skips the negative forward.

Only *decision tokens* are scored: the DPO term masks out formatting scaffolding so the
gradient lands on the tool choice and arguments, not on the boilerplate the expert and
negative share.

### Training schedule

| Stage | Steps | `sft_weight` | `dpo_weight` | Negatives |
|---|---|---|---|---|
| **R0** — SFT warm-up | 75 | 0.5 | 0.0 | none |
| **R1 … R5** — refresh rounds | 75 each | 0.5 | 1.0 | regenerated each round from the previous round's merged checkpoint |

Total 450 optimizer steps at effective batch size 16 (`4 GPUs × bs 1 × grad_accum 4`), i.e.
1,200 pair visits per round. Each round writes `negatives_lookup_r{R}.json` keyed by
`f"{pair_id}__{variant}"`, then merges its LoRA adapter to become the next round's sampler.

---

## Repository layout

```
.
├── data/
│   └── step_pairs_canonical.json      # canonical STB step pairs (15.7 MB)
├── src/
│   ├── data/
│   │   └── ppa_render.py              # PPA renderer registry
│   └── training/
│       ├── sft_lora.py                # model loading + LoRA config
│       ├── agentic_dpo_trainer.py     # loss, collator, online-PPA dataset
│       └── agentic_dpo_negative_gen.py# vLLM K-sample negative generator
└── scripts/
    ├── build_stb_multi_schema.py      # STB renderer (VARIANTS + build_variant)
    ├── neggen_online_ppa.py           # per-round negative refresh
    ├── merge_lora.py                  # merge adapter into base checkpoint
    └── run_stb_agentic_dpo.sh         # full R0 → R5 pipeline
```

### Data format

`data/step_pairs_canonical.json` is a JSON list of canonical step pairs:

| Field | Description |
|---|---|
| `pair_id` | stable identifier, `{task_id}_step{idx}` |
| `state_messages` | chat-template messages for the state (system + history) |
| `expert_step_text` | full expert assistant turn (Thought + Action + Action Input) |
| `expert_action_text` | action only — used for the DPO decision-token mask |
| `expert_tool_name` | gold tool name |
| `expert_action_input` | gold tool arguments |

Variant tags, log-probs and `negative_step_text` are filled in at runtime: the renderer
suffixes `pair_id` with the chosen variant and clears stale negative-side fields, and the
negative is resolved from the lookup written by `scripts/neggen_online_ppa.py`.

---

## Hyper-parameters

Recipe values are hard-coded in `scripts/run_stb_agentic_dpo.sh` for reproducibility.

| Flag | Value | Role |
|---|---|---|
| `--beta` | `0.008` | DPO temperature |
| `--alpha` | `0.5` | length-scale exponent, `β_eff = β / max_len^α` |
| `--sft_weight` | `0.5` | SFT anchor coefficient (load-bearing) |
| `--dpo_weight` | `0.0` → `1.0` | off in R0, on from R1 |
| `--lora_rank` / `--lora_alpha` | `64` / `128` | LoRA capacity |
| `--quantization` | `int4` | NF4 4-bit base, bf16 LoRA |
| `--lr` | `2e-4` | cosine schedule, 3% warm-up |
| `--max_seq_length` | `2048` | crop long contexts |
| `--online_ppa_domain` | `stb` | enable on-the-fly variant rendering |
| `K` (neggen) | `4` | candidates per state; `K ≥ 4` recommended (paper §4.4) |
| `STEPS_PER_ROUND` | `75` | optimizer steps per round |
| `N_DPO_ROUNDS` | `5` | refresh rounds R1 … R5 |
| `GRAD_ACC` | `4` | effective batch size = `NUM_GPUS × 1 × 4` |

The trainer also exposes `--loss_type` variants (`cnll`, `ipo`, `listwise`, `gated_hinge`,
`grpo_step`) used in the paper's ablations. Leave the default (`dpo`) for the standard recipe.

---

## Adding a new environment

`src/data/ppa_render.py` is a registry. A renderer module needs exactly two symbols:

```python
VARIANTS = ("_base", "_json", "_rename", "_combined")   # your Φ

def build_variant(pair: dict, variant: str) -> dict:
    """Deep-copy `pair` and rewrite it consistently under `variant`."""
```

`build_variant` must rewrite the system prompt's tool list, every prior assistant turn, the
expert action/step text and `expert_tool_name` — all consistently, so the pair stays
self-consistent under the new schema.

**It must also be a pure function of `(pair, variant)`.** The negatives lookup is keyed by
`f"{pair_id}__{variant}"`, so a renderer that uses unseeded randomness will silently
mismatch negatives to states. `scripts/build_stb_multi_schema.py` seeds its per-pair RNG from
`blake2b(pair_id)` for exactly this reason; use it as the reference implementation.

Then register it and pass `--online_ppa_domain <your_domain>`.

---

## Citation

```bibtex
@article{chen2026agenticdpo,
  title   = {Agentic-DPO: From Imitation to Agentic Policy Optimization on Expert Trajectories},
  author  = {Chen, Yixiong and Yuille, Alan},
  journal = {arXiv preprint arXiv:2607.10601},
  year    = {2026}
}
```

## License

Released under the [MIT License](LICENSE).

`data/step_pairs_canonical.json` is derived from ToolBench trajectories; please also observe
the [ToolBench](https://github.com/OpenBMB/ToolBench) license terms when redistributing it.

## Acknowledgements

Built on [StableToolBench](https://github.com/THUNLP-MT/StableToolBench),
[ToolBench](https://github.com/OpenBMB/ToolBench), [PEFT](https://github.com/huggingface/peft)
and [vLLM](https://github.com/vllm-project/vllm).
