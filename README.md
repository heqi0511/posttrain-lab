# Posttrain Lab

Posttrain Lab is a small research repo for post-training approximately 5B language models.
The first path is supervised fine-tuning, focused on format, domain behavior, and tool-use protocol.
After SFT is stable, RLVR/GRPO will be used to improve answer correctness with deterministic verifiers.
TRL is the MVP framework for both SFT and early RLVR experiments.
verl is a later migration target only if TRL becomes a bottleneck after data, reward, and eval are proven.
Data curation, reward design, and eval regression are kept as separate responsibilities.
Smoke tests and overfit-32 checks come before larger training runs.
Eval prompts, metrics, labels, and baselines are treated as frozen unless reviewed.
Every experiment should write reproducible run cards and resolved configs.
The current repository contains scaffolding only, not full training implementations.

