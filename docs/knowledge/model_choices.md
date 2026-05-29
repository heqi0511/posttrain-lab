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

This is a small gate experiment, not a model-quality result. Larger SFT runs require a passing overfit-32 check and explicit approval.
