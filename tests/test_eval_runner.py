import json
import subprocess
import sys
from pathlib import Path

import pytest

from posttrain_lab.eval.eval_runner import run_eval
from posttrain_lab.eval.metrics import exact_match, format_success


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def test_metric_helpers_are_strict_and_deterministic():
    assert exact_match(" 42 ", "42")
    assert not exact_match("42.", "42")
    assert format_success(r"^\\boxed\{.+\}$", r"\boxed{42}")
    assert not format_success(r"^\\boxed\{.+\}$", "42")


def test_dry_run_eval_writes_generations_metrics_and_report(tmp_path):
    prompt_path = tmp_path / "prompts.jsonl"
    output_dir = tmp_path / "eval_run"
    write_jsonl(
        prompt_path,
        [
            {
                "id": "ex-1",
                "prompt": "Return boxed answer for 2 + 2.",
                "answer": r"\boxed{4}",
                "mock_generation": r"\boxed{4}",
            },
            {
                "id": "ex-2",
                "prompt": "Return boxed answer for 3 + 3.",
                "answer": r"\boxed{6}",
                "mock_generation": "6",
            },
        ],
    )

    metrics = run_eval(
        {
            "prompt_path": str(prompt_path),
            "output_dir": str(output_dir),
            "dry_run": True,
            "model_name": "dummy",
            "inference": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": 16,
                "stop_tokens": [],
            },
            "metrics": {
                "exact_match": True,
                "format_regex": r"^\\boxed\{.+\}$",
            },
        }
    )

    assert metrics["count"] == 2
    assert metrics["exact_match"] == 0.5
    assert metrics["format_success"] == 0.5

    generations = read_jsonl(output_dir / "raw_generations.jsonl")
    assert generations[0]["id"] == "ex-1"
    assert generations[0]["generation"] == r"\boxed{4}"
    assert generations[1]["format_success"] is False

    metrics_json = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_json == metrics

    report = (output_dir / "eval_report.md").read_text(encoding="utf-8")
    assert "Eval Report" in report
    assert "exact_match" in report
    assert "format_success" in report


def test_boxed_math_match_scores_final_answer_without_exact_text_match(tmp_path):
    prompt_path = tmp_path / "prompts.jsonl"
    output_dir = tmp_path / "eval_run"
    write_jsonl(
        prompt_path,
        [
            {
                "id": "math-1",
                "prompt": "Solve 2 + 2.",
                "answer": "4",
                "mock_generation": "We compute 2 + 2 = 4, so the final answer is \\boxed{4}.",
            },
            {
                "id": "math-2",
                "prompt": "Solve 3 + 3.",
                "answer": "6",
                "mock_generation": "The answer is 6.",
            },
        ],
    )

    metrics = run_eval(
        {
            "prompt_path": str(prompt_path),
            "output_dir": str(output_dir),
            "dry_run": True,
            "model_name": "dummy",
            "inference": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": 16,
                "stop_tokens": [],
            },
            "metrics": {
                "exact_match": False,
                "boxed_math_match": True,
                "allow_symbolic_equivalence": False,
                "format_regex": None,
            },
        }
    )

    assert metrics["answer_match"] == 0.5
    assert metrics["answer_parse_failure_rate"] == 0.5
    assert metrics["parse_failure_rate"] == 0.5

    generations = read_jsonl(output_dir / "raw_generations.jsonl")
    assert generations[0]["answer_match"] is True
    assert generations[0]["parsed_answer"] == "4"
    assert generations[1]["answer_failure_reason"] == "no_boxed_answer"


def test_boxed_math_match_can_use_sympy_equivalence_engine(tmp_path):
    pytest.importorskip("latex2sympy2")
    prompt_path = tmp_path / "prompts.jsonl"
    output_dir = tmp_path / "eval_run"
    write_jsonl(
        prompt_path,
        [
            {
                "id": "sympy-1",
                "prompt": "Expand 2(x + 1).",
                "answer": "2x+2",
                "mock_generation": r"\boxed{2(x+1)}",
            }
        ],
    )

    metrics = run_eval(
        {
            "prompt_path": str(prompt_path),
            "output_dir": str(output_dir),
            "dry_run": True,
            "model_name": "dummy",
            "inference": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": 16,
                "stop_tokens": [],
            },
            "metrics": {
                "exact_match": False,
                "boxed_math_match": True,
                "allow_symbolic_equivalence": True,
                "symbolic_equivalence_engine": "sympy",
                "format_regex": None,
            },
        }
    )

    assert metrics["answer_match"] == 1.0
    assert metrics["answer_parse_failure_rate"] == 0.0
    generations = read_jsonl(output_dir / "raw_generations.jsonl")
    assert generations[0]["answer_match"] is True
    assert generations[0]["answer_failure_reason"] is None


def test_stop_tokens_are_applied_in_dry_run(tmp_path):
    prompt_path = tmp_path / "prompts.jsonl"
    output_dir = tmp_path / "eval_run"
    write_jsonl(
        prompt_path,
        [
            {
                "id": "stop-1",
                "prompt": "Return before stop token.",
                "answer": "final",
                "mock_generation": "final<END>extra text",
            }
        ],
    )

    run_eval(
        {
            "prompt_path": str(prompt_path),
            "output_dir": str(output_dir),
            "dry_run": True,
            "model_name": "dummy",
            "inference": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": 16,
                "stop_tokens": ["<END>"],
            },
            "metrics": {"exact_match": True},
        }
    )

    generation = read_jsonl(output_dir / "raw_generations.jsonl")[0]["generation"]
    assert generation == "final"


def test_cli_runs_from_config_in_dry_run(tmp_path):
    prompt_path = tmp_path / "prompts.jsonl"
    output_dir = tmp_path / "eval_run"
    config_path = tmp_path / "baseline.yaml"
    write_jsonl(
        prompt_path,
        [
            {
                "id": "cli-1",
                "prompt": "Return 10.",
                "answer": "10",
                "mock_generation": "10",
            }
        ],
    )
    config_path.write_text(
        "\n".join(
            [
                f"prompt_path: {prompt_path}",
                f"output_dir: {output_dir}",
                "dry_run: true",
                "model_name: dummy",
                "inference:",
                "  temperature: 0.0",
                "  top_p: 1.0",
                "  max_new_tokens: 8",
                "  stop_tokens: []",
                "metrics:",
                "  exact_match: true",
                "  format_regex: null",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "posttrain_lab.eval.eval_runner",
            "--config",
            str(config_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "raw_generations.jsonl").exists()
    assert "exact_match" in result.stdout
