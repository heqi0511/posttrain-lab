import json
from pathlib import Path

from posttrain_lab.train.train_grpo import (
    load_config,
    load_rlvr_train_examples,
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
                "  max_completion_length: 16",
                "  temperature: 0.7",
                "  top_p: 0.95",
                "  top_k: 0",
                "  beta: 0.0",
                "  sample_count: 4",
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
    assert config["selection"]["max_train_examples"] == 4
    assert config["rollout"]["num_generations"] == 2
    assert config["rollout"]["max_completion_length"] == 16


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
