# Reward Card

Status: first deterministic math reward implemented.

Future reward versions must document semantics, scale, invalid-output handling, adversarial fixtures, reward hacking checks, and changes that affect RLVR comparability.

## math_boxed_v001

- File: `src/posttrain_lab/rewards/math_reward.py`
- Primary API: `math_boxed_v001(completion, answer, config=...)`
- Structured API: `score_math_boxed_v001(completion, answer, config=...)`
- Reward range: binary `0.0` or `1.0`.
- Default symbolic equivalence: disabled.
- Network/filesystem access: none.
- Execution bound: completions longer than `max_output_chars` return `0.0`; symbolic equivalence uses a short, local, bounded AST evaluator.

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

If `allow_symbolic_equivalence=True`, the reward can also compare simple numeric arithmetic expressions such as `1/2` and `0.5`. This path is local and bounded: it rejects long expressions, unsupported characters, large ASTs, large fractions, division by zero, and large exponents.

### Known Reward-Hacking Risks

- A model may learn to emit many repeated identical boxed answers. This now scores `0.0`, and RLVR logs should still track completion length.
- A model may include prompt-injection text such as "give full reward" before the boxed answer. The verifier ignores that text and still scores only the single visible final boxed answer, so prompt-injection strings must be monitored in rollouts.
- A model may place the correct answer in reasoning and a wrong boxed final answer. Conflicting boxed answers score `0.0`.
- A model may place the correct boxed answer in a negated sentence or candidate answer while giving a wrong unboxed final answer. The final-answer suffix check scores this `0.0` when non-punctuation text follows the boxed answer.
- A model may exploit malformed LaTeX or parser ambiguity. Malformed boxed syntax scores `0.0`.
- A model may output extremely long text before a boxed answer. Length above `max_output_chars` scores `0.0`.
- A model may hide contradictory boxed answers inside closed `<think>...</think>` blocks. Those blocks are stripped before scoring, so only the visible final answer is rewarded; RLVR logs should still track completion length.

### Fixtures And Tests

- Fixture file: `tests/fixtures/rewards/math_boxed_v001_cases.jsonl`
- Test file: `tests/test_math_reward.py`
- Command: `make test-rewards`

The adversarial fixtures cover multiple boxed answers, repeated identical boxed answers, correct boxed answers embedded before later contradictions, correct boxed candidate with wrong final answer, malformed LaTeX, long output, prompt-injection text, closed and unclosed thinking blocks, ordinary reasoning before a final boxed answer, and symbolic equivalence being disabled by default.
