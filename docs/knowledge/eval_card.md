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
