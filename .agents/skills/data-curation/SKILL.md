---
name: data-curation
description: Use when staging, validating, deduplicating, filtering, splitting, or auditing SFT/RLVR datasets, JSONL records, prompts, answer fields, leakage, or data cards.
---

# Data Curation

## When to Use

Use for dataset schema changes, JSONL validation, deduplication, filtering, split audits, RLVR prompt staging, reward fixtures, and data-card updates.

Do not use for reward semantics, eval metric design, or training loop changes.

## Required Workflow

1. Identify source, license, intended split, and target task family.
2. Treat `data/raw/` as read-only unless the user explicitly permits editing it.
3. Validate schema before filtering or deduplication.
4. Check leakage against eval, validation, hidden tests, and reward fixtures.
5. Write staged data only under `data/staged/`, `data/train/`, `data/val/`, `data/test/`, `data/rlvr_prompts/`, or `data/reward_fixtures/`.
6. Update `docs/knowledge/data_card.md` for schema, filtering, split, and hash changes.
7. Request human review before changing split logic or adding new training datasets.

## Required Commands

```bash
make validate-data
make check-leakage
```

If available for the file type:

```bash
python .agents/skills/data-curation/scripts/validate_sft_jsonl.py <path>
python .agents/skills/data-curation/scripts/dedup.py <input> <output>
```

## Invariants / Forbidden Actions

- Never edit `data/raw/` without explicit instruction.
- Never mix eval or hidden-test answers into train data.
- Never change splits without data-card updates and review.
- Never silently drop hard examples, failures, or minority formats.
- Never rely on ad hoc string parsing when a schema exists.

## Common Failure Modes

- Schema drift between SFT records, RLVR prompts, and eval examples.
- Near-duplicate train examples leaking into validation or eval.
- Filtering that removes failures and inflates apparent quality.
- Prompt/answer fields swapped or partially normalized.
- Missing data hashes, source metadata, or split provenance.
- Hidden tests copied into training or reward fixtures.

## References

- `docs/knowledge/data_card.md`
- `docs/knowledge/failure_taxonomy.md`
- `docs/knowledge/experiment_log.md`
- `.agents/skills/data-curation/references/data_schema.md`
