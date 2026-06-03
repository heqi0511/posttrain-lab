import json
from pathlib import Path

from posttrain_lab.train.train_grpo import (
    _build_grpo_config_kwargs,
    _math_reward_config,
    _rollout_format_gate_passed,
    _resolve_config,
    _summarize_samples,
    _summarize_trainer_signal,
    load_config,
    load_rlvr_train_examples,
    make_math_boxed_reward_func,
    math_boxed_reward_func,
    run_grpo,
)


def write_rlvr_jsonl(path, count):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            left = index + 1
            right = index + 2
            record = {
                "id": f"rlvr-train-{index:03d}",
                "split": "train",
                "prompt": [
                    {
                        "role": "user",
                        "content": f"Compute {left} + {right}. Return only the final answer in boxed format.",
                    }
                ],
                "verifier": {"type": "math_boxed_v001", "answer": str(left + right)},
                "metadata": {
                    "source": "unit-test",
                    "domain": "math",
                    "difficulty": "easy",
                    "license": "synthetic",
                },
            }
            handle.write(json.dumps(record) + "\n")


def write_config(path, data_path, output_dir):
    path.write_text(
        "\n".join(
            [
                "run_name: grpo-smoke-test",
                "model_name_or_path: hf-internal-testing/tiny-random-gpt2",
                "adapter_path: null",
                f"data_path: {data_path}",
                f"output_dir: {output_dir}",
                "dry_run: true",
                "synthetic_data_if_missing: false",
                "smoke_run: true",
                "seed: 17",
                "torch_dtype: auto",
                "trust_remote_code: false",
                "enable_thinking: false",
                "reward_version: math_boxed_v001",
                "selection:",
                "  train_split: train",
                "  max_train_examples: 4",
                "training:",
                "  max_steps: 1",
                "  per_device_train_batch_size: 2",
                "  gradient_accumulation_steps: 1",
                "  learning_rate: 0.000001",
                "rollout:",
                "  num_generations: 2",
                "  max_completion_length: 32",
                "  temperature: 0.7",
                "  top_p: 0.95",
                "  top_k: 0",
                "  beta: 0.0",
                "  sample_count: 4",
                "rollout_format_gate:",
                "  enabled: true",
                "  sample_count: 4",
                "  max_parse_failure_rate: 0.0",
                "peft:",
                "  method: lora",
                "  r: 4",
                "  lora_alpha: 8",
                "  lora_dropout: 0.0",
                "  target_modules: []",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_load_config_parses_grpo_smoke_fields(tmp_path):
    data_path = tmp_path / "rlvr.jsonl"
    output_dir = tmp_path / "runs" / "rlvr" / "grpo_smoke"
    config_path = tmp_path / "grpo_smoke.yaml"
    write_config(config_path, data_path, output_dir)

    config = load_config(config_path)

    assert config["run_name"] == "grpo-smoke-test"
    assert config["reward_version"] == "math_boxed_v001"
    assert config["adapter_path"] is None
    assert config["selection"]["max_train_examples"] == 4
    assert config["rollout"]["num_generations"] == 2
    assert config["rollout"]["max_completion_length"] == 32
    assert config["rollout_format_gate"]["enabled"] is True
    assert config["rollout_format_gate"]["max_parse_failure_rate"] == 0.0


def test_qwen3_smoke_config_uses_sft_adapter_and_format_gate():
    config = load_config("configs/rlvr/qwen3_0_6b_grpo_smoke.yaml")

    assert config["adapter_path"] == "runs/sft/smoke_1k_boxed"
    assert config["rollout"]["max_completion_length"] == 32
    assert config["rollout_format_gate"]["enabled"] is True
    assert config["rollout_format_gate"]["max_parse_failure_rate"] == 0.0
    assert "sft_init" in config["output_dir"]


def test_frontier_grpo_smoke_config_records_requested_sampling_and_early_stop():
    config = load_config("configs/rlvr/frontier_grpo_smoke.yaml")

    assert config["rollout"]["num_generations"] == 8
    assert config["rollout"]["temperature"] == 0.9
    assert config["rollout"]["top_p"] == 0.95
    assert config["rollout"]["max_completion_length"] == 384
    assert config["rollout"]["sample_rollout_generations"] == 8
    assert config["training"]["frac_reward_zero_std_early_stop"] is True
    assert config["training"]["frac_reward_zero_std_threshold"] == 0.8
    assert config["training"]["frac_reward_zero_std_patience"] == 20


def test_qwen25_dapo_grpo_config_matches_paperish_hyperparams():
    config = load_config("configs/rlvr/qwen25_math_1_5b_dapo_grpo_paperish.yaml")
    resolved = _resolve_config(config)

    assert resolved["model_name_or_path"].endswith("Qwen2.5-Math-1.5B")
    assert resolved["reward_version"] == "math_boxed_verl_v001"
    assert resolved["selection"]["max_train_examples"] == 17917
    assert resolved["training"]["per_device_train_batch_size"] == 16
    assert resolved["training"]["learning_rate"] == 0.000002
    assert resolved["training"]["max_steps"] == 280
    assert resolved["rollout"]["num_generations"] == 4
    assert resolved["rollout"]["max_prompt_length"] == 1024
    assert resolved["rollout"]["max_completion_length"] == 2048
    assert resolved["rollout"]["temperature"] == 0.8
    assert resolved["rollout"]["beta"] == 0.0
    assert resolved["rollout"]["epsilon"] == 0.22
    assert resolved["rollout"]["num_iterations"] == 2
    assert resolved["rollout"]["loss_type"] == "dapo"
    assert resolved["rollout"]["scale_rewards"] == "none"
    assert resolved["peft"]["method"] == "none"
    assert resolved["reward"]["allow_symbolic_equivalence"] is True


def test_build_grpo_config_kwargs_passes_optional_trl_fields(tmp_path):
    config = _resolve_config(load_config("configs/rlvr/qwen25_math_1_5b_dapo_grpo_paperish.yaml"))
    supported = {
        "output_dir",
        "max_steps",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "logging_steps",
        "logging_first_step",
        "save_steps",
        "save_total_limit",
        "save_strategy",
        "report_to",
        "seed",
        "bf16",
        "fp16",
        "gradient_checkpointing",
        "gradient_checkpointing_kwargs",
        "num_generations",
        "max_completion_length",
        "temperature",
        "top_p",
        "top_k",
        "beta",
        "epsilon",
        "epsilon_high",
        "num_iterations",
        "loss_type",
        "scale_rewards",
        "steps_per_generation",
        "ds3_gather_for_generation",
        "mask_truncated_completions",
        "use_vllm",
        "vllm_mode",
        "vllm_gpu_memory_utilization",
        "vllm_max_model_length",
        "optim",
        "lr_scheduler_type",
        "warmup_ratio",
        "max_grad_norm",
        "dataloader_num_workers",
        "remove_unused_columns",
        "log_completions",
        "num_completions_to_print",
        "model_init_kwargs",
        "chat_template_kwargs",
    }

    kwargs = _build_grpo_config_kwargs(config, tmp_path, supported_fields=supported)

    assert kwargs["per_device_train_batch_size"] == 16
    assert kwargs["num_generations"] == 4
    assert kwargs["max_completion_length"] == 2048
    assert kwargs["epsilon"] == 0.22
    assert kwargs["num_iterations"] == 2
    assert kwargs["scale_rewards"] == "none"
    assert kwargs["loss_type"] == "dapo"
    assert "gradient_checkpointing_kwargs" not in kwargs
    assert "max_prompt_length" not in kwargs


def test_resolve_config_defaults_reward_equivalence_off(tmp_path):
    data_path = tmp_path / "rlvr.jsonl"
    output_dir = tmp_path / "runs" / "rlvr" / "grpo_smoke"
    config_path = tmp_path / "grpo_smoke.yaml"
    write_config(config_path, data_path, output_dir)

    resolved = _resolve_config(load_config(config_path))
    reward_config = _math_reward_config(resolved)

    assert reward_config.allow_symbolic_equivalence is False
    assert reward_config.symbolic_equivalence_engine == "fraction"


def test_resolve_config_accepts_explicit_sympy_reward_engine(tmp_path):
    data_path = tmp_path / "rlvr.jsonl"
    output_dir = tmp_path / "runs" / "rlvr" / "grpo_smoke"
    config_path = tmp_path / "grpo_smoke.yaml"
    write_config(config_path, data_path, output_dir)
    with config_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n".join(
                [
                    "reward:",
                    "  allow_symbolic_equivalence: true",
                    "  symbolic_equivalence_engine: sympy",
                    "  max_symbolic_expr_chars: 200",
                    "  max_symbolic_ast_nodes: 128",
                    "  max_symbolic_collection_size: 16",
                ]
            )
            + "\n"
        )

    resolved = _resolve_config(load_config(config_path))
    reward_config = _math_reward_config(resolved)

    assert reward_config.allow_symbolic_equivalence is True
    assert reward_config.symbolic_equivalence_engine == "sympy"
    assert reward_config.max_symbolic_expr_chars == 200
    assert reward_config.max_symbolic_ast_nodes == 128
    assert reward_config.max_symbolic_collection_size == 16


def test_load_rlvr_train_examples_selects_exact_train_records(tmp_path):
    data_path = tmp_path / "rlvr.jsonl"
    write_rlvr_jsonl(data_path, 6)

    examples = load_rlvr_train_examples(data_path, split="train", limit=4)

    assert len(examples) == 4
    assert examples[0]["id"] == "rlvr-train-000"
    assert examples[-1]["verifier"]["answer"] == "9"


def test_math_boxed_reward_func_scores_batch_answers():
    rewards = math_boxed_reward_func(
        prompts=["p1", "p2"],
        completions=[r"\boxed{4}", r"\boxed{5}"],
        answer=["4", "4"],
    )

    assert rewards == [1.0, 0.0]


def test_configured_reward_func_can_use_common_verl_style_reward():
    reward_func = make_math_boxed_reward_func(_math_reward_config({}), "math_boxed_verl_v001")

    rewards = reward_func(
        prompts=["p", "p"],
        completions=[r"The answer is \boxed{4}. extra text", r"The answer is \boxed{5}."],
        answer=["4", "4"],
    )

    assert rewards == [1.0, 0.1]


def test_rollout_format_gate_rejects_saturated_rewards():
    config = {
        "rollout_format_gate": {
            "max_parse_failure_rate": 0.0,
            "max_reward_mean": 0.95,
            "max_perfect_reward_rate": 0.95,
        }
    }

    assert not _rollout_format_gate_passed(
        config,
        {
            "parse_failure_rate": 0.0,
            "reward_mean": 1.0,
            "perfect_reward_rate": 1.0,
        },
    )
    assert _rollout_format_gate_passed(
        config,
        {
            "parse_failure_rate": 0.0,
            "reward_mean": 0.75,
            "perfect_reward_rate": 0.75,
        },
    )


def test_summarize_samples_reports_frontier_group_signal():
    metrics = _summarize_samples(
        [
            {
                "reward_vector": [0.0, 1.0, 0.0, 1.0],
                "parse_failure_vector": [False, False, False, False],
                "completion_lengths": [10, 12, 14, 16],
            },
            {
                "reward_vector": [1.0, 1.0, 1.0, 1.0],
                "parse_failure_vector": [False, False, False, False],
                "completion_lengths": [8, 8, 8, 8],
            },
        ]
    )

    assert metrics["reward_mean"] == 0.75
    assert metrics["frac_reward_zero_std"] == 0.5
    assert metrics["effective_mixed_group_rate"] == 0.5
    assert metrics["parse_failure_rate"] == 0.0


def test_summarize_trainer_signal_reports_nonzero_grad_on_mixed_steps():
    metrics = _summarize_trainer_signal(
        [
            {
                "step": 1,
                "frac_reward_zero_std": 1.0,
                "reward": 0.0,
                "reward_std": 0.0,
                "grad_norm": 0.0,
                "completions/mean_length": 40,
            },
            {
                "step": 2,
                "frac_reward_zero_std": 0.0,
                "reward": 0.5,
                "reward_std": 0.5,
                "grad_norm": 0.25,
                "completions/mean_length": 52,
            },
        ]
    )

    assert metrics["frac_reward_zero_std"] == 0.5
    assert metrics["effective_mixed_group_rate"] == 0.5
    assert metrics["nonzero_grad_step_rate"] == 0.5
    assert metrics["avg_completion_length"] == 46


def test_dry_run_grpo_writes_required_artifacts(tmp_path):
    data_path = tmp_path / "rlvr.jsonl"
    output_dir = tmp_path / "runs" / "rlvr" / "grpo_smoke"
    config_path = tmp_path / "grpo_smoke.yaml"
    write_rlvr_jsonl(data_path, 4)
    write_config(config_path, data_path, output_dir)

    result = run_grpo(load_config(config_path), config_path=config_path)

    assert result["reward_mean"] == 1.0
    assert result["reward_std"] == 0.0
    assert result["zero_reward_rate"] == 0.0
    assert result["perfect_reward_rate"] == 1.0
    assert result["parse_failure_rate"] == 0.0
    assert result["avg_completion_length"] > 0
    assert (output_dir / "resolved_config.yaml").exists()
    assert (output_dir / "run_card.md").exists()
    assert (output_dir / "metrics.jsonl").exists()
    assert (output_dir / "sample_rollouts.jsonl").exists()
    assert (output_dir / "rollout_format_gate.json").exists()
    assert (output_dir / "rollout_format_gate.jsonl").exists()

    sample_rollout = json.loads((output_dir / "sample_rollouts.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert set(sample_rollout) >= {
        "prompt",
        "completion",
        "reward",
        "parsed_answer",
        "failure_reason",
    }
    assert sample_rollout["parsed_answer"] == sample_rollout["answer"]
    assert sample_rollout["failure_reason"] is None

    run_card = (output_dir / "run_card.md").read_text(encoding="utf-8")
    assert "reward version: `math_boxed_v001`" in run_card
    assert "adapter path: `None`" in run_card
    assert "rollout format gate parse failure rate: `0.0`" in run_card
    assert "parse failure rate: `0.0`" in run_card
    assert "avg completion length:" in run_card


def test_dry_run_can_create_synthetic_rlvr_data(tmp_path):
    data_path = tmp_path / "missing" / "rlvr.jsonl"
    output_dir = tmp_path / "runs" / "rlvr" / "grpo_smoke"
    config_path = tmp_path / "grpo_smoke.yaml"
    write_config(config_path, data_path, output_dir)
    text = config_path.read_text(encoding="utf-8")
    text = text.replace("synthetic_data_if_missing: false", "synthetic_data_if_missing: true")
    config_path.write_text(text, encoding="utf-8")

    run_grpo(load_config(config_path), config_path=config_path)

    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    assert rows[0]["verifier"]["type"] == "math_boxed_v001"
    assert "boxed format" in rows[0]["prompt"][0]["content"]
