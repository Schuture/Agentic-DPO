#!/bin/bash
# Agentic-DPO on StableToolBench (STB), 2B backbone — paper recipe with
# **online PPA** and **R = 5 refresh rounds** (Algorithm 1 in the paper).
#
# Pipeline:
#   R0   SFT warm-up. The trainer reads canonical pairs and samples a schema
#        variant phi ~ Unif(Phi) for every gradient step; no negative needed
#        (sft_weight=0.5, dpo_weight=0). 75 optimizer steps.
#   for r in 1 .. 5:
#       N_r  Refresh negatives. Render every (canonical_pair × variant) under
#            the registered PPA renderer, run vLLM K-sample (K=4, no fallback)
#            against the current student, write a fresh
#                negatives_lookup_r${r}.json
#            keyed by f"{pair_id}__{variant}".
#       D_r  Agentic-DPO with online PPA + the round-r lookup. The trainer
#            re-samples a variant per step, looks up the matching negative,
#            and trains 75 optimizer steps with sft_weight=0.5, dpo_weight=1.0.
#       The merged checkpoint of round r becomes the sampler for round r+1.
#
# Hyper-parameters (paper §4 main table; same across backbones / benchmarks):
#   β = 0.008, α = 0.5, λ = 0.5, K = 4, effective batch size = 16
#   (NUM_GPUS=4, per_device_bs=1, ga=4 ⇒ eff_bs=16). Length-scaled implicit
#   reward margin β_eff = β / max(|a⁺|,|a⁻|)^α. Δ_ref ≡ 0 (frozen-zero ref;
#   stable under small β + SFT anchor — see paper App. C.2).
#
# Required environment variables:
#   SEED            random seed (e.g. 42).
#   BASE_MODEL      path to the Qwen3.5-2B base checkpoint.
#   CANON_PAIRS     canonical step-pair JSON (provided in data/).
#   OUTPUT_DIR      directory to hold intermediate ckpts and the final merged model.
#   PYTHON_TRAIN    python with peft / transformers / bitsandbytes installed.
#   PYTHON_EVAL     python with vllm installed (used only for neggen).
#   TORCHRUN        torchrun binary from the training env.
#   NUM_GPUS        DDP world size (default 4).
#
# Usage:
#   bash scripts/run_stb_agentic_dpo.sh
#   # or with SLURM:
#   sbatch --export=ALL,SEED=42 scripts/run_stb_agentic_dpo.sh

set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: "${SEED:=42}"
: "${BASE_MODEL:?must set BASE_MODEL to the Qwen3.5-2B checkpoint path}"
: "${CANON_PAIRS:=data/step_pairs_canonical.json}"
: "${OUTPUT_DIR:=./runs/stb_agentic_dpo_s${SEED}}"
: "${PYTHON_TRAIN:=python}"
: "${PYTHON_EVAL:=python}"
: "${TORCHRUN:=torchrun}"
: "${NUM_GPUS:=4}"

WARMUP_STEPS=75
N_DPO_ROUNDS=5
STEPS_PER_ROUND=75
K_NEG=4                 # candidates per state — paper §4.4 requires K ≥ 4
MAX_SEQ_LEN=2048
GRAD_ACC=4              # NUM_GPUS=4 × per_device_bs=1 × ga=4 ⇒ eff_bs=16

ZERO_REF="${OUTPUT_DIR}/ref_logprobs_zero.json"

mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Agentic-DPO STB 2B (online PPA, R=${N_DPO_ROUNDS})  seed=${SEED}"
echo "  Canonical pairs:   ${CANON_PAIRS}"
echo "  Base model:        ${BASE_MODEL}"
echo "  Output dir:        ${OUTPUT_DIR}"
echo "  R0 warm-up steps:  ${WARMUP_STEPS}"
echo "  Rounds × steps:    ${N_DPO_ROUNDS} × ${STEPS_PER_ROUND}"
echo "  Neggen K:          ${K_NEG} (no fallback)"
echo "  β=0.008  α=0.5  λ(sft_weight)=0.5  max_seq=${MAX_SEQ_LEN}"
echo "  LoRA r=64 / α=128, lr=2e-4, int4 quant"
echo "  Effective batch:   ${NUM_GPUS} × 1 × ${GRAD_ACC} = $((NUM_GPUS * GRAD_ACC))"
echo "=========================================="

# Build a zero-ref cache once. The OnlinePPAStepDataset tolerates missing keys
# and falls back to Δ_ref = 0; this file is here only to satisfy the trainer's
# --ref_cache CLI contract.
${PYTHON_TRAIN} - <<PY
import json
pairs = json.load(open("${CANON_PAIRS}"))
ref = {p['pair_id']: {'expert': 0.0, 'negative': 0.0} for p in pairs if 'pair_id' in p}
json.dump(ref, open("${ZERO_REF}", "w"))
print(f"[init] zero-ref cache: {len(ref)} entries  ->  ${ZERO_REF}")
PY

EFF_BS=$((NUM_GPUS * GRAD_ACC))                # = 16
N_VISITS_PER_ROUND=$((STEPS_PER_ROUND * EFF_BS))   # = 1200 — strict round budget

train_one_round () {
    # train_one_round R BASE STEP_PAIRS OUT_LORA OUT_MERGED DPO_W [LOOKUP]
    local R="$1" BASE="$2" STEP_PAIRS="$3" OUT_LORA="$4" OUT_MERGED="$5" DPO_W="$6"
    local LOOKUP_ARG=""
    if [ -n "${7:-}" ]; then
        LOOKUP_ARG="--negatives_lookup $7"
    fi
    echo ""; echo ">>> R${R} train  dpo_weight=${DPO_W}  base=$(basename "${BASE}")  step_pairs=$(basename "${STEP_PAIRS}")  $(date)"
    ${TORCHRUN} --nproc_per_node=${NUM_GPUS} --master_port=$((29500 + (SEED + R*29 + 53) % 1000)) \
        -m src.training.agentic_dpo_trainer \
        --model_path "${BASE}" --model_name qwen3.5-2b \
        --step_pairs "${STEP_PAIRS}" --ref_cache "${ZERO_REF}" --output_dir "${OUT_LORA}" \
        --online_ppa_domain stb ${LOOKUP_ARG} \
        --beta 0.008 --alpha 0.5 \
        --sft_weight 0.5 --dpo_weight ${DPO_W} \
        --epochs 1 --max_steps ${STEPS_PER_ROUND} \
        --batch_size 1 --gradient_accumulation_steps ${GRAD_ACC} --lr 2e-4 \
        --max_seq_length ${MAX_SEQ_LEN} --quantization int4 --lora_rank 64 --lora_alpha 128 \
        --seed $((SEED + R))
    local AP="${OUT_LORA}"
    [ -f "${AP}/adapter_config.json" ] || AP=$(ls -d "${OUT_LORA}"/checkpoint-* | sort -t- -k2 -n | tail -1)
    ${PYTHON_TRAIN} scripts/merge_lora.py --base_model "${BASE}" --adapter "${AP}" --output "${OUT_MERGED}"
}

refresh_negatives_lookup () {
    # refresh_negatives_lookup R SAMPLER LOOKUP_OUT SUBSET_PAIRS_OUT
    local R="$1" SAMPLER="$2" LOOKUP_OUT="$3" SUBSET_OUT="$4"
    echo ""; echo ">>> R${R} neggen via $(basename "${SAMPLER}")  budget=${N_VISITS_PER_ROUND} (strict)  $(date)"
    CUDA_VISIBLE_DEVICES=0 ${PYTHON_EVAL} scripts/neggen_online_ppa.py \
        --canonical_pairs "${CANON_PAIRS}" \
        --domain stb \
        --sampler_model "${SAMPLER}" \
        --lookup_out "${LOOKUP_OUT}" \
        --subset_pairs_out "${SUBSET_OUT}" \
        --n_visits ${N_VISITS_PER_ROUND} \
        --round_seed $((SEED * 100 + R)) \
        --K ${K_NEG} --temperature 0.7 --max_tokens 256 --max_model_len 4096
}

# ── R0 — SFT warm-up over canonical × Φ (variant sampled per step) ──
# R0 has no negatives lookup (SFT-only fallback path of OnlinePPAStepDataset
# returns weight=0 on every DPO term). It iterates the FULL canonical pool;
# the round budget only constrains R1..R5.
WARMUP_LORA="${OUTPUT_DIR}/r0_warmup_sft"
WARMUP_MERGED="${OUTPUT_DIR}/r0_warmup_merged"
train_one_round 0 "${BASE_MODEL}" "${CANON_PAIRS}" "${WARMUP_LORA}" "${WARMUP_MERGED}" 0.0
CUR_BASE="${WARMUP_MERGED}"

# ── R1 .. R5 — per-round neggen (strict budget) + Agentic-DPO ──
for R in $(seq 1 ${N_DPO_ROUNDS}); do
    LOOKUP="${OUTPUT_DIR}/negatives_lookup_r${R}.json"
    SUBSET="${OUTPUT_DIR}/subset_pairs_r${R}.json"
    OUT_LORA="${OUTPUT_DIR}/r${R}_dpo"
    OUT_MERGED="${OUTPUT_DIR}/r${R}_dpo_merged"

    refresh_negatives_lookup ${R} "${CUR_BASE}" "${LOOKUP}" "${SUBSET}"
    train_one_round ${R} "${CUR_BASE}" "${SUBSET}" "${OUT_LORA}" "${OUT_MERGED}" 1.0 "${LOOKUP}"
    CUR_BASE="${OUT_MERGED}"

    # Free disk: drop the previous round's LoRA folder (the merged checkpoint
    # is what gets reused as the next round's sampler).
    if [ ${R} -gt 1 ]; then
        rm -rf "${OUTPUT_DIR}/r$((R-1))_dpo" 2>/dev/null || true
    fi
done

echo ""
echo "=========================================="
echo "Done. Per-round merged checkpoints:"
ls -d "${OUTPUT_DIR}"/r*_dpo_merged "${OUTPUT_DIR}/r0_warmup_merged" 2>/dev/null
echo ""
echo "Pick the best round on a held-out validation slice (paper §6.4)."
echo "=========================================="
