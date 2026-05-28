import csv
import json

from posttrain_lab.data.rollout_audit import load_config, run_rollout_audit
from posttrain_lab.data.validate import validate_file


def write_rlvr_jsonl(path, count):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            record = {
                "id": f"frontier-{index:03d}",
                "split": "train",
                "prompt": [
                    {
                        "role": "user",
                        "content": f"Compute {index + 2} + {index + 3}. Return only the final answer in boxed format.",
                    }
                ],
                "verifier": {"type": "math_boxed_v001", "answer": str(2 * index + 5)},
                "metadata": {
                    "source": "unit-test",
                    "domain": "math",
                    "difficulty": "medium",
                    "license": "synthetic",
                },
            }
            handle.write(json.dumps(record) + "\n")


def write_config(path, input_path, output_dir, filtered_path, excluded_path):
    path.write_text(
        "\n".join(
            [
                f"input_path: {input_path}",
                f"output_dir: {output_dir}",
                f"filtered_output_path: {filtered_path}",
                f"excluded_output_path: {excluded_path}",
                "dry_run: true",
                "model_name_or_path: Qwen/Qwen3-0.6B",
                "adapter_path: null",
                "seed: 31",
                "torch_dtype: auto",
                "trust_remote_code: false",
                "enable_thinking: false",
                "reward_version: math_boxed_v001",
                "selection:",
                "  split: train",
                "  max_prompts: 10",
                "rollout:",
                "  completions_per_prompt: 16",
                "  max_new_tokens: 32",
                "  temperature: 0.7",
                "  top_p: 0.95",
                "  top_k: 0",
                "frontier:",
                "  min_reward_mean: 0.2",
                "  max_reward_mean: 0.8",
                "  max_parse_failure_rate: 0.5",
                "  min_unique_answer_count: 3",
                "review:",
                "  max_prompts: 5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_rollout_audit_writes_frontier_artifacts_without_training(tmp_path):
    input_path = tmp_path / "rlvr.jsonl"
    output_dir = tmp_path / "audit"
    filtered_path = tmp_path / "frontier_grpo_train.jsonl"
    excluded_path = tmp_path / "frontier_excluded.jsonl"
    config_path = tmp_path / "audit.yaml"
    write_rlvr_jsonl(input_path, 10)
    write_config(config_path, input_path, output_dir, filtered_path, excluded_path)

    summary = run_rollout_audit(load_config(config_path), config_path=config_path)

    assert summary["no_training_executed"] is True
    assert summary["audited_prompt_count"] == 10
    assert summary["target_prompt_count"] == 10
    assert summary["completed"] is True
    assert summary["completions_per_prompt"] == 16
    assert summary["generation_batch_size"] == 8
    assert summary["bucket_counts"] == {"all_zero": 4, "all_one": 2, "mixed": 4}
    assert summary["all_zero_count"] == 4
    assert summary["all_one_count"] == 2
    assert summary["mixed_count"] == 4
    assert summary["all_zero_rate"] == 0.4
    assert summary["all_one_rate"] == 0.2
    assert summary["mixed_rate"] == 0.4
    assert summary["parse_failure_rate"] == 0.2
    assert summary["unique_answer_count"] > 0
    assert summary["avg_completion_length"] > 0
    assert summary["kept_prompt_count"] == 2
    assert summary["effective_mixed_group_rate"] == 0.4
    assert summary["exclude_reason_counts"] == {
        "all_zero": 2,
        "all_one": 2,
        "low_diversity": 2,
        "parse_fail": 2,
    }

    assert (output_dir / "rollout_audit_summary.json").exists()
    assert (output_dir / "rollout_audit_by_prompt.csv").exists()
    assert (output_dir / "sample_rollouts_for_review.jsonl").exists()
    assert filtered_path.exists()
    assert excluded_path.exists()

    report = json.loads((output_dir / "rollout_audit_summary.json").read_text(encoding="utf-8"))
    assert report["effective_mixed_group_rate"] == 0.4

    with (output_dir / "rollout_audit_by_prompt.csv").open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["bucket"] == "mixed"
    assert rows[0]["task_type"] == "math"
    assert rows[0]["keep"] == "True"

    filtered_report = validate_file(filtered_path, "rlvr")
    assert filtered_report.ok, filtered_report.errors
    assert filtered_report.num_rows == 2

    review_rows = (output_dir / "sample_rollouts_for_review.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(review_rows) == 5 * 16
