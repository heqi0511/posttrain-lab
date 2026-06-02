# Eval Card

Status: baseline dry-run eval runner implemented.

The eval runner reads JSONL prompts, generates outputs, writes `raw_generations.jsonl`, `metrics.json`, and `eval_report.md`, and supports exact-match, regex-based format-success, and boxed-answer math matching metrics.

## Format Success

For LaTeX boxed-answer format checks, the intended regex is a literal `\boxed{...}` match. YAML configs should prefer:

```yaml
format_regex: ^\\boxed\{.+\}$
```

The eval helper also normalizes the common single-backslash pattern `^\boxed\{.+\}$` to the same literal boxed-answer check, because Python regex otherwise interprets `\b` as a word boundary. This normalization is a reporting bug fix only; it does not change reward semantics, eval prompts, labels, decoding settings, or answer correctness.

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

## Sampled Heldout Eval

Status: sampled heldout eval implemented for checking whether GRPO changes the model distribution, not just greedy/top-1 output.

Command shape:

```bash
python -m posttrain_lab.eval.sampled_eval --config <sampled_eval.yaml>
```

The sampled eval reads the same heldout JSONL format as the greedy eval, samples `K` completions per prompt, and writes `sampled_generations.jsonl`, `sampled_by_prompt.jsonl`, `metrics.json`, and `eval_report.md`.

Required reported metrics:

- `sampled_accuracy`: mean reward over all sampled completions.
- `pass_at_K`: fraction of prompts with at least one correct sampled completion.
- `all_correct_rate`, `all_zero_rate`, and `mixed_prompt_rate`.
- `format_success_rate`, `parse_failure_rate`, and average completion length.

Comparability rule: sampled evals are comparable only when they use the same heldout file, `K`, seed, temperature, top-p, max-new-token budget, reward version, and symbolic-equivalence settings.

## OpenR1-Style Reasoning Eval

For reasoning SFT runs trained on `<think>...</think>` traces, use a separate reasoning eval instead of changing the fixed final-only baseline.

- Enable long generation budgets such as `max_new_tokens: 2048` for smoke runs.
- Set `enable_thinking: true` when evaluating Qwen3/OpenR1-style reasoning behavior.
- Score final answers with `boxed_math_match`, which extracts boxed answers from the completion and compares them against the reference answer.
- Keep `exact_match` and strict final-only regex checks for format-contract evals; do not use them as the main math-quality metric for reasoning runs.
- Report `answer_match`, `answer_parse_failure_rate`, `parse_failure_rate`, and completion length alongside raw generations.

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

Local dataset smoke shape:

```bash
python -m posttrain_lab.eval.math_dataset_eval \
  --dataset-id local-rlvr-fixture \
  --dataset-path tests/fixtures/rlvr_good.jsonl \
  --dataset-format jsonl \
  --model-name dummy \
  --output-dir /tmp/posttrain_lab_qwen25_math_eval \
  --prompt-template paper_math \
  --dry-run
```

Evaluation policy:

- Use a fixed seed and record `sample_size`, dataset id/config/split, model id, decoding settings, and thinking mode.
- For local paper-style evals, record `dataset_path`, `dataset_format`, and `prompt_template`; use `paper_math` only as an explicit run setting, not as a silent replacement for existing baselines.
- Score with the existing `math_boxed_v001` reward; do not change reward semantics for dataset comparisons.
- Save `raw_generations.jsonl`, `sample_generations.jsonl`, `eval_summary.json`, and `eval_report.md`.
- Treat accuracy as strict boxed-answer accuracy. It is useful for RLVR readiness but may undercount mathematically equivalent answers not normalized by the current reward.
- Report parse failure rate, truncation rate, and correctness given parse so format failures are not confused with math failures.
- Nexus runs should use `scripts/slurm/run_math_dataset_eval.sh` from a clean git commit.
