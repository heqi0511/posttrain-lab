# Data Card

Status: strict schema validators implemented for SFT and RLVR JSONL fixtures.

No real datasets are included yet. Raw data must remain read-only unless explicitly reviewed. Future updates should record source, license, schema, split logic, hashes, filters, deduplication, and leakage checks.

## Common Rules

- Files are UTF-8 JSONL with exactly one JSON object per line.
- Required `split` values are `train`, `val`, or `test`.
- `id` values must be non-empty strings and unique within a file.
- `metadata` must contain non-empty string fields: `source`, `domain`, `difficulty`, and `license`.
- Empty message or prompt content is invalid.
- Validation errors report file path and line number.

## SFT JSONL Schema

Required top-level fields: `id`, `split`, `messages`, `metadata`.

`messages` must be a non-empty list of objects with exactly `role` and `content`. Valid roles are `system`, `user`, and `assistant`; at least one assistant message is required.

Example:

```json
{"id":"sft-001","split":"train","messages":[{"role":"system","content":"You are careful."},{"role":"user","content":"Compute 2 + 2."},{"role":"assistant","content":"4"}],"metadata":{"source":"unit-test","domain":"math","difficulty":"easy","license":"synthetic"}}
```

## RLVR JSONL Schema

Required top-level fields: `id`, `split`, `prompt`, `verifier`, `metadata`.

`prompt` must be a non-empty list of user-only messages with exactly `role` and `content`. Assistant messages in RLVR prompts are invalid.

`verifier` must contain non-empty string fields: `type` and `answer`.

Example:

```json
{"id":"rlvr-001","split":"train","prompt":[{"role":"user","content":"Compute 5 + 7."}],"verifier":{"type":"math_boxed_v001","answer":"12"},"metadata":{"source":"unit-test","domain":"math","difficulty":"easy","license":"synthetic"}}
```

## Validation Commands

```bash
python -m posttrain_lab.data.validate --type sft --path <file>
python -m posttrain_lab.data.validate --type rlvr --path <file>
make validate-data
```
