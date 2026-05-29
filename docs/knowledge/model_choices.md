# Model Choices

## Qwen3-0.6B

- Hugging Face model id: `Qwen/Qwen3-0.6B`
- Current role: first real SFT overfit-32 target after local tiny-model pipeline checks.
- Framework: TRL `1.5.1` with PEFT LoRA.
- Precision: `bfloat16` on Nexus GPU nodes.
- Adapter strategy: LoRA on attention and MLP projection modules.
- Context length for the first overfit gate: `128` tokens.
- Run config: `configs/sft/qwen3_0_6b_overfit32.yaml`
- OpenR1 math SFT config: `configs/sft/openr1_math_1k.yaml`
- OpenR1 long-context smoke config: `configs/sft/openr1_math_1k_len8192.yaml`
- Long-context smoke policy: start with `8192` training tokens and `2048` eval/sample generation tokens before considering `16384` or `32768`.

This is a small gate experiment, not a model-quality result. Larger SFT runs require a passing overfit-32 check and explicit approval.

## Qwen3-4B

- Hugging Face model id: `Qwen/Qwen3-4B`
- Current role: small short-boxed SFT and GRPO-feasibility checks after Qwen3-0.6B produced too many all-zero reward groups.
- Framework: TRL `1.5.1` with PEFT LoRA.
- Precision: `bfloat16` on Nexus GPU nodes.
- Adapter strategy: LoRA on attention and MLP projection modules.
- Initial short-boxed SFT config: `configs/sft/qwen3_4b_openr1_format_repair_tiny.yaml`
- Scale policy: keep the first run small (`512` train examples, `128` validation examples, `300` steps), select the checkpoint with minimum validation loss, then run a tiny no-training rollout audit before any GRPO trainer step.

This is a capability probe for GRPO readiness, not a final model-quality run.
