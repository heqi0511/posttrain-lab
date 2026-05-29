import json

import pytest

from posttrain_lab.eval.math_dataset_eval import (
    extract_answer,
    extract_prompt,
    format_math_prompt,
    normalize_dataset_record,
    run_math_dataset_eval,
    summarize_eval_rows,
)


def test_extract_deepmath_prompt_and_solution_answer():
    record = {
        "prompt": [{"role": "user", "content": "Compute 2+2."}],
        "solution": "$4$",
    }

    assert extract_prompt(record) == "Compute 2+2."
    assert extract_answer(record) == "$4$"

    example = normalize_dataset_record(record, dataset_id="trl-lib/DeepMath-103K", index=3)
    assert example.id == "trl-lib__DeepMath-103K-000003"
    assert example.answer == "$4$"


def test_extract_openr1_problem_and_answer_metadata():
    record = {
        "uuid": "abc",
        "problem": "Find x if x + 3 = 8.",
        "answer": "5",
        "subject": "Algebra",
        "type": "math-word-problem",
        "source": "olympiads",
    }

    example = normalize_dataset_record(record, dataset_id="open-r1/OpenR1-Math-220k", index=0)

    assert example.id == "abc"
    assert example.prompt == "Find x if x + 3 = 8."
    assert example.answer == "5"
    assert example.metadata["subject"] == "Algebra"
    assert example.metadata["source"] == "olympiads"


def test_format_prompt_requires_single_final_boxed_answer():
    prompt = format_math_prompt("Compute 1+1.")

    assert "exactly one \\boxed{...}" in prompt
    assert "Do not write anything after the boxed answer." in prompt
    assert "Compute 1+1." in prompt


def test_summarize_eval_rows_reports_accuracy_parse_and_truncation():
    rows = [
        {
            "reward": 1.0,
            "reason": "exact_match",
            "parse_failed": False,
            "completion_chars": 10,
            "completion_tokens": 3,
            "truncated": False,
            "metadata": {"subject": "Algebra"},
        },
        {
            "reward": 0.0,
            "reason": "answer_mismatch",
            "parse_failed": False,
            "completion_chars": 20,
            "completion_tokens": 6,
            "truncated": True,
            "metadata": {"subject": "Algebra"},
        },
        {
            "reward": 0.0,
            "reason": "no_boxed_answer",
            "parse_failed": True,
            "completion_chars": 30,
            "completion_tokens": 9,
            "truncated": False,
            "metadata": {"subject": "Geometry"},
        },
    ]

    summary = summarize_eval_rows(rows, {"dataset_id": "d", "model_name": "m", "sample_size": 3})

    assert summary["accuracy"] == pytest.approx(1 / 3)
    assert summary["parse_failure_rate"] == pytest.approx(1 / 3)
    assert summary["format_success_rate"] == pytest.approx(2 / 3)
    assert summary["correctness_given_parse"] == 0.5
    assert summary["truncation_rate"] == 1 / 3
    assert summary["reason_counts"] == {"answer_mismatch": 1, "exact_match": 1, "no_boxed_answer": 1}


def test_dry_run_math_dataset_eval_writes_artifacts(monkeypatch, tmp_path):
    examples = [
        normalize_dataset_record(
            {"problem": "Compute 0.", "answer": "0", "id": "ex-0"},
            dataset_id="dummy",
            index=0,
        )
    ]
    monkeypatch.setattr("posttrain_lab.eval.math_dataset_eval.load_eval_examples", lambda **_: examples)

    summary = run_math_dataset_eval(
        {
            "dataset_id": "dummy",
            "dataset_config": "default",
            "split": "train",
            "model_name": "dummy-model",
            "output_dir": str(tmp_path),
            "sample_size": 1,
            "seed": 7,
            "shuffle_buffer_size": 1,
            "max_new_tokens": 16,
            "batch_size": 1,
            "dry_run": True,
            "enable_thinking": "false",
            "temperature": 0.0,
            "top_p": 1.0,
            "save_sample_count": 1,
        }
    )

    assert summary["accuracy"] == 1.0
    assert (tmp_path / "eval_summary.json").exists()
    rows = [json.loads(line) for line in (tmp_path / "raw_generations.jsonl").read_text().splitlines()]
    assert rows[0]["completion"] == r"\boxed{0}"
