# Failure Taxonomy

Status: active for SFT and RLVR comparison reports.

Use this document to track recurring data, formatting, reward, eval, and RLVR failure modes. Keep reward hacking, parse failures, length drift, leakage, and eval regression categories separate.

## RLVR Heldout Failure Categories

Use these categories when extracting failures after SFT+RLVR eval:

- `wrong reasoning`: the final answer is wrong and the correct answer is not otherwise evident in the generation.
- `correct reasoning wrong final`: the generation contains the correct answer or a correct intermediate result, but the final parsed answer is wrong.
- `parser failure`: the output cannot be parsed by the boxed-answer parser because the boxed expression is missing, empty, malformed, or not final-only.
- `multiple answers`: the output contains more than one boxed answer, including repeated identical boxed answers.
- `too long`: the output exceeds the comparison tool's configured length threshold or the reward parser reports `output_too_long`.
- `format violation`: the output violates the required eval format even when it may contain useful text.

## Reporting Rules

- Do not drop failed examples from comparison reports.
- Extract up to 50 SFT+RLVR failures into `failure_cases.jsonl`.
- Report category counts in `comparison_report.md`.
- Do not change eval prompts, reward semantics, or answer extraction to reclassify failures.
- If no failures are found, keep the empty failure file and report zero counts explicitly.
