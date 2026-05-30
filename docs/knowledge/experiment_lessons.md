# Experiment Lessons

Status: active working summary.

This document summarizes the main problems we hit while building the SFT -> RLVR/GRPO workflow, the evidence we saw, and the fixes that changed later decisions. It is intentionally selective; detailed run-by-run records stay in `docs/knowledge/experiment_log.md`.

## 1. Format Mismatch Can Hide Real Progress

Problem: early SFT runs taught plain numeric answers, while the fixed eval prompt expected a final boxed answer. This produced misleading metrics: the model could learn arithmetic, but eval reported `target_eval_score = 0.0` because outputs were not in the required `\boxed{...}` format.

Evidence:

- The first Qwen3-0.6B smoke-1k run reached low train loss, but eval-after-train had exact match `0.0`, format success `0.0`, and parse failure `1.0`.
- The later boxed-format smoke generated `\boxed{4}` for `2 + 2` and reached exact match `1.0`, format success `1.0`, parse failure `0.0`.

Fix:

- Make the SFT target match the reward/eval contract: exactly one final boxed answer.
- Add eval-after-train and sample generations to catch contract mismatches immediately.
- Treat format repair as a separate phase from math ability improvement.

Current lesson: before judging model quality, first verify that training data, reward parser, and eval parser agree on the same answer format.

## 2. Qwen Thinking Mode And Truncation Caused Many Parse Failures

Problem: OpenR1-style reasoning and Qwen thinking mode often generated long answers with unclosed `<think>` blocks or no visible final boxed answer under short token budgets.

Evidence:

- GSM8K Qwen3-4B scout with `max_new_tokens=128` had parse failure rate about `0.56`.
- Parse-failure taxonomy found major categories: `truncated_before_final`, `no_boxed_answer`, and `final_answer_unboxed`.
- OpenR1 long-context 0.6B SFT evaluation still had `7/8` generations with `unclosed_think_block` under `max_new_tokens=2048`.

Fix:

- Added prompt/generation sweeps to separate truncation, prompt wording, and thinking-mode effects.
- Updated `math_boxed_v001` to ignore complete `<think>...</think>` content, while still rejecting unclosed think blocks.
- Used short boxed / format-repair SFT with thinking disabled for concise RLVR-ready outputs.

Current lesson: long reasoning is useful for capability, but for the first RLVR loop we need a short, parseable answer interface. Format stability is a gate, not a secondary detail.

## 3. Reward Strictness Prevented Easy Reward Hacking, But Exposed Data Issues

Problem: a loose reward parser can be gamed by multiple answers, answers hidden in reasoning, malformed boxes, or prompt-injection text. A strict reward avoids that, but it can under-credit mathematically equivalent multi-solution formats.

Evidence:

- We tightened `math_boxed_v001` to require exactly one visible `\boxed{...}` answer; repeated identical boxed answers and multiple conflicting boxed answers both receive reward `0`.
- Adversarial tests covered multiple boxed answers, wrong final answer after correct reasoning, malformed LaTeX, extremely long output, and prompt injection.
- In the checkpoint-5250 audit, answers like `\sqrt{3}` or `\pm\sqrt{3}` could be mathematically related to the target but still received `0` when the expected target format was a set or assignment.

Fix:

- Added optional SymPy + latex2sympy2 equivalence behind an explicit config flag.
- Kept reward semantics deterministic and test-covered.
- Filtered the training pool to `single_expression_no_assignment_v1`, rejecting assignment/equation targets and plus-minus style multi-answer targets for the current phase.

Current lesson: strict reward is the right default for early RLVR because it reduces reward hacking. The cost is narrower data eligibility, so data curation must match the verifier's real capabilities.

## 4. GRPO Needs Within-Prompt Reward Variance, Not Just Aggregate Reward Variance

Problem: several GRPO runs had nonzero average reward variance across all samples, but most prompt groups were all wrong or all correct. In GRPO, those groups have zero useful advantage.

Evidence:

- Synthetic expanded run `6916959` showed many steps with `frac_reward_zero_std=1.0`, `grad_norm=0.0`, and loss `0.0`.
- Qwen3-0.6B SFT-init smoke was format-perfect and reward-perfect, but reward std was `0.0`, so there was no learning signal.
- Later runs logged `frac_reward_zero_std`, `effective_mixed_group_rate`, and `nonzero_grad_step_rate`, which made this failure mode visible.

Fix:

- Added rollout-format gates before real GRPO.
- Added metrics for `frac_reward_zero_std`, `effective_mixed_group_rate`, `parse_failure_rate`, and completion length.
- Used small rollout audits to check whether a checkpoint and dataset can produce mixed reward vectors before scaling.

Current lesson: "reward_mean went up" is not enough. A GRPO run is only useful when enough prompt groups contain both correct and incorrect completions.

## 5. Frontier Audit Helps Signal, But Can Be Too Expensive

Problem: offline frontier selection can identify useful prompts, but K completions per prompt on larger models and datasets adds substantial generation cost.

Evidence:

- Early GSM8K audit jobs were cancelled because the first implementation generated too slowly.
- Batched generation improved runtime, but large audits were still too expensive to run by default.
- On synthetic frontier prompts, effective mixed group rate improved from about `0.28` audit mixed rate to `0.97` during a frontier GRPO smoke, proving the method works when used selectively.

Fix:

- Batched audit generation with multiple return sequences.
- Added partial audit outputs so cancelled jobs still leave inspectable data.
- Disabled frontier audit by default; targets now require `RUN_FRONTIER_AUDIT=1`.
- Preferred micro audits before any larger GRPO run.

Current lesson: frontier selection is a diagnostic and data-selection tool, not a default preprocessing step. Use it sparingly, then decide whether the signal gain justifies the compute.

## 6. Dataset Difficulty Must Match The Current Policy

Problem: toy synthetic data was too easy, GSM8K was often too easy for Qwen3-4B once formatting was controlled, while OpenR1/DeepMath was much harder and exposed real math limitations.

Evidence:

- Toy SFT quickly fixed format and easy arithmetic, but RLVR had little room to improve.
- GSM8K Qwen3-4B parseable completions were often correct, but parse failures and all-one/all-zero groups made it a weak main GRPO source.
- Pilot eval showed harder datasets were meaningfully challenging: Qwen3-4B strict boxed accuracy was `0.36` on DeepMath-103K samples and `0.26` on OpenR1-Math-220k samples.

Fix:

- Moved toward OpenR1/DeepMath-derived data.
- Built an OpenR1 CN math pool from `cn_k12`, `cn_contest`, and `amc_aime`, limited to Algebra and Number Theory.
- Added staged RLVR/SFT datasets and preserved split policy instead of touching `data/raw`.

Current lesson: useful RLVR data lives near the policy frontier. Too easy gives all-one groups; too hard gives all-zero groups; incompatible answer formats give parse failures.

## 7. SFT Improved Format And Basic Accuracy, But Did Not Solve The Task Alone

Problem: SFT was necessary to make outputs parseable, but more SFT did not automatically translate into large heldout math accuracy gains.

Evidence:

- Qwen3-4B format-repair SFT reduced parse failures to `0.0`, but accuracy on harder OpenR1 validation remained limited.
- The OpenR1 CN math SFT selected `checkpoint-5250` by validation loss, with heldout generation accuracy around `0.375` on a 64-prompt validation generation check.
- Validation loss plateaued after roughly step `4500`, so blindly continuing SFT was unlikely to be the best next move.

Fix:

- Selected checkpoints by validation loss rather than final step by habit.
- Used SFT as a format and domain warmup, then audited rollout reward signal before GRPO.
- Avoided treating low parse failure as proof of strong math ability.

Current lesson: for this project, SFT is mainly making a clean policy interface and improving baseline competence. Heldout accuracy still has to be measured directly.

## 8. Eval Bugs And Metric Ambiguity Need Independent Checks

Problem: a metric can be wrong even when reward parsing is correct. In one GRPO comparison, `format_success` was reported as `0` because the regex treated `\b` in `\boxed` as a word boundary.

Evidence:

- `math_boxed_v001` showed zero parse failures, while the eval runner's format success field reported failure.
- Rechecking format success on saved generations gave `256/256 = 1.0` before and after GRPO.

Fix:

- Added a format-success recheck and corrected the reporting path.
- Kept eval prompts and reward semantics unchanged while debugging the metric.
- Added sampled heldout eval so we can observe distribution shifts, not just greedy top-1.

Current lesson: never change prompts or reward to make metrics look better. First inspect saved generations and verify the metric implementation.

## 9. GRPO Scaling Produced Small But Real Distribution Shifts

Problem: after format and reward signal were stable, GRPO still produced only small heldout gains. More steps did not create a large greedy accuracy jump.

Evidence:

- Small GRPO: heldout accuracy `0.3125 -> 0.3242` on 256 prompts.
- Medium GRPO: heldout accuracy `0.3125 -> 0.3184` on 512 prompts.
- Continuation to 2000 total steps: greedy heldout `0.3125 -> 0.3164`, while sampled accuracy improved `0.2632 -> 0.2830` and pass@8 improved `0.5020 -> 0.5098`.
- Greedy changes were small: `7` wrong-to-correct, `5` correct-to-wrong, and most generations unchanged.

Fix:

- Added frozen before/after heldout comparison.
- Added sampled heldout eval with pass@K, mixed prompt rate, and unique answer count.
- Preserved checkpoints during larger runs so future selection does not depend only on the final adapter.

Current lesson: current GRPO setup is technically healthy, but the model-quality gain is modest. The next improvement likely needs better prompt selection, data difficulty balance, or checkpoint-level selection rather than simply more steps.

## 10. Operational Discipline Prevented Invalid Results

Problem: server-side experiment work can easily drift from local source code, and failed jobs can leave misleading partial artifacts.

Evidence:

- Several server jobs failed due to OOM, quoting/schema issues, or slow generation, but run logs and partial outputs made the cause clear.
- The project adopted local edit -> local checks -> commit/push -> server checkout as the standard path.
- Run cards record git commit, data hashes, config hashes, server paths, and caveats.

Fix:

- Treat local git as the source of truth.
- Keep generated `runs/` artifacts out of source control unless explicitly requested.
- Record failed runs and failed setup attempts instead of deleting them.

Current lesson: reproducibility work feels slow, but it prevented us from confusing code drift, data drift, metric bugs, and true model behavior.

## Current Working Conclusions

- Qwen3-0.6B is useful for cheap smoke tests, but it has not been a strong direct GRPO policy on harder math.
- Qwen3-4B plus concise boxed SFT is the current practical policy for small RLVR experiments.
- `math_boxed_v001` with optional SymPy equivalence is good enough for single-expression boxed answers, but not for broad multi-solution algebra without more verifier design.
- The single-expression OpenR1 CN math pool is currently the cleanest RLVR source because it matches the reward parser better.
- Frontier audit should remain optional because it is expensive, but micro audits are valuable before training.
- The next meaningful progress should come from better data/checkpoint selection and stronger heldout analysis, not from simply extending the same GRPO run.
