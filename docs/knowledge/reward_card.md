# Reward Card

Status: first deterministic math reward implemented.

Future reward versions must document semantics, scale, invalid-output handling, adversarial fixtures, reward hacking checks, and changes that affect RLVR comparability.

## math_boxed_v001

- File: `src/posttrain_lab/rewards/math_reward.py`
- Primary API: `math_boxed_v001(completion, answer, config=...)`
- Structured API: `score_math_boxed_v001(completion, answer, config=...)`
- Reward range: binary `0.0` or `1.0`.
- Default symbolic equivalence: disabled.
- Optional symbolic engines: `fraction` for the existing bounded numeric arithmetic checker, or `sympy` for `latex2sympy2` + SymPy expression equivalence.
- Network/filesystem access: none.
- Execution bound: completions longer than `max_output_chars` return `0.0`; symbolic equivalence uses local bounded parsing and rejects long expressions, large expression trees, unsupported syntax, and large collections.

### Contract

The completion must contain exactly one well-formed visible boxed final answer using `\boxed{...}`. The boxed answer may be preceded by ordinary reasoning, but after the boxed answer only math delimiters, punctuation, or whitespace are allowed. The reward normalizes simple formatting and returns `1.0` only when that single normalized boxed answer matches the normalized reference answer.

Invalid outputs receive `0.0`, including:

- no boxed answer;
- malformed `\boxed{...}` syntax;
- unclosed `<think>` block;
- empty boxed answer;
- multiple boxed answers, even when repeated identical;
- a boxed answer embedded before later non-punctuation text, negation, or a contradictory final answer;
- answer mismatch;
- output longer than the configured limit.

Repeated identical boxed answers are rejected to remove parser ambiguity and reduce reward-hacking surface.

### Parser Behavior

Before boxed-answer extraction, the parser removes complete `<think>...</think>` blocks and then applies the same exactly-one-visible-boxed final-answer check to the remaining visible completion. Unclosed thinking blocks are not removed and do not get special treatment.

Normalization removes whitespace and simple LaTeX noise, including `\left`, `\right`, math delimiters, and simple `\frac{a}{b}` / `\frac12` forms. Exact normalized string match is checked first.

If `allow_symbolic_equivalence=True`, non-exact equivalence is enabled explicitly:

- `symbolic_equivalence_engine: fraction` preserves the older bounded numeric arithmetic checker for cases such as `1/2` and `0.5`.
- `symbolic_equivalence_engine: sympy` uses `latex2sympy2` plus SymPy to compare parser-compatible final expressions, including examples such as `1/2` versus `0.5`, `2(x+1)` versus `2x+2`, and unordered finite sets such as `\{1,2,3\}` versus `\{3,2,1\}`.

The SymPy path is off by default. It rejects equations containing `=`, natural-language tokens, `\text{...}`, environments, unsupported characters, expressions longer than the configured limit, expression trees larger than the configured node cap, and collections larger than the configured collection cap. This is intentional because `latex2sympy2` can otherwise parse equation-like strings such as `x=1` too loosely for reward use.

### Known Reward-Hacking Risks

- A model may learn to emit many repeated identical boxed answers. This now scores `0.0`, and RLVR logs should still track completion length.
- A model may include prompt-injection text such as "give full reward" before the boxed answer. The verifier ignores that text and still scores only the single visible final boxed answer, so prompt-injection strings must be monitored in rollouts.
- A model may place the correct answer in reasoning and a wrong boxed final answer. Conflicting boxed answers score `0.0`.
- A model may place the correct boxed answer in a negated sentence or candidate answer while giving a wrong unboxed final answer. The final-answer suffix check scores this `0.0` when non-punctuation text follows the boxed answer.
- A model may exploit malformed LaTeX or parser ambiguity. Malformed boxed syntax scores `0.0`.
- A model may exploit a symbolic parser by writing equation-like or prose-like answers. The optional SymPy engine rejects `=`, text macros, natural-language tokens, and unsupported syntax before parsing.
- A model may output extremely long text before a boxed answer. Length above `max_output_chars` scores `0.0`.
- A model may hide contradictory boxed answers inside closed `<think>...</think>` blocks. Those blocks are stripped before scoring, so only the visible final answer is rewarded; RLVR logs should still track completion length.

### Fixtures And Tests

- Fixture file: `tests/fixtures/rewards/math_boxed_v001_cases.jsonl`
- Test file: `tests/test_math_reward.py`
- Command: `make test-rewards`

The adversarial fixtures cover multiple boxed answers, repeated identical boxed answers, correct boxed answers embedded before later contradictions, correct boxed candidate with wrong final answer, malformed LaTeX, long output, prompt-injection text, closed and unclosed thinking blocks, ordinary reasoning before a final boxed answer, symbolic equivalence being disabled by default, optional SymPy equivalence, wrong algebra under SymPy, unordered set comparison, and equation-parser hacking.

## math_boxed_verl_v001

- File: `src/posttrain_lab/rewards/math_reward.py`
- verl entrypoint: `src/posttrain_lab/rewards/verl_math_reward.py`
- Function name for verl configs: `compute_score_verl_style` or alias `compute_score_common`
- Reward range: `0.0`, `0.1`, or `1.0`.
- Default symbolic equivalence: disabled, with the same optional engines as `math_boxed_v001`.

### Contract

This is a common-verl-style diagnostic/training reward, not a replacement for the strict verifier. It extracts the last well-formed visible `\boxed{...}` answer after stripping complete `<think>...</think>` blocks.

Scoring:

- `1.0`: the extracted boxed answer matches the reference by exact normalization or explicitly enabled symbolic equivalence.
- `0.1`: at least one well-formed boxed answer exists, but the final extracted answer is wrong.
- `0.0`: no boxed answer, malformed boxed syntax, empty boxed answer, unclosed `<think>` block, overlong output, missing ground truth, or overlong reference.

Unlike `math_boxed_v001`, this version does not require exactly one boxed answer and does not require the boxed answer to be the last visible text. It follows the looser pattern commonly used in verl/TRL examples where format correctness can receive partial credit.

### Known Reward-Hacking Risks

- A model can receive `0.1` for format-only behavior even when the answer is wrong.
- Multiple boxed answers are allowed; the last boxed answer is used. This is easier to optimize but weaker against contradictory-answer exploits.
- Text after the boxed answer does not invalidate a correct answer, so verbose or self-contradictory continuations must be monitored separately through completion length and rollout samples.
- This reward should be logged as `math_boxed_verl_v001` and compared separately from strict `math_boxed_v001` runs.
