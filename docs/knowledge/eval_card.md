# Eval Card

Status: baseline dry-run eval runner implemented.

The eval runner reads JSONL prompts, generates outputs, writes `raw_generations.jsonl`, `metrics.json`, and `eval_report.md`, and supports exact-match plus regex-based format-success metrics.

## JSONL Input

Each eval record must include:

- `id`: stable example identifier
- `prompt`: prompt string or message list

Optional fields:

- `answer`: target for exact-match scoring
- `mock_generation`: dry-run generation for deterministic tests and smoke runs

## Inference Config

Eval runs must record fixed inference settings:

- `temperature`
- `top_p`
- `max_new_tokens`
- `stop_tokens`

Dry-run mode uses `mock_generation` and does not load a model. Non-dry-run mode uses a local Hugging Face causal LM model path/name.

## Comparability Rules

- Do not change prompts, labels, answer extraction, decoding settings, filters, metrics, or aggregation to improve reported results.
- Compare runs only when they use the same eval config and metric definitions.
- If an eval bug is fixed, record the new eval version and do not compare it directly to older results without calling out the version change.
- Always preserve raw generations for manual review.

## Baseline Command

```bash
make eval-baseline
```

## Math Dataset Pilot Eval

Status: pilot evaluator implemented for sampled Hugging Face math datasets.

Command shape:

```bash
python -m posttrain_lab.eval.math_dataset_eval \
  --dataset-id trl-lib/DeepMath-103K \
  --dataset-config default \
  --split train \
  --model-name Qwen/Qwen3-4B \
  --output-dir runs/eval/math_dataset_pilot/qwen3_4b_deepmath \
  --sample-size 50 \
  --seed 20260529 \
  --max-new-tokens 2048 \
  --enable-thinking false \
  --trust-remote-code
```

Evaluation policy:

- Use a fixed seed and record `sample_size`, dataset id/config/split, model id, decoding settings, and thinking mode.
- Score with the existing `math_boxed_v001` reward; do not change reward semantics for dataset comparisons.
- Save `raw_generations.jsonl`, `sample_generations.jsonl`, `eval_summary.json`, and `eval_report.md`.
- Treat accuracy as strict boxed-answer accuracy. It is useful for RLVR readiness but may undercount mathematically equivalent answers not normalized by the current reward.
- Report parse failure rate, truncation rate, and correctness given parse so format failures are not confused with math failures.
- Nexus runs should use `scripts/slurm/run_math_dataset_eval.sh` from a clean git commit.
