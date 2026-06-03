import json

import pytest

from posttrain_lab.eval.math_dataset_eval import (
    extract_answer,
    extract_prompt,
    format_math_prompt,
    load_eval_examples,
    normalize_dataset_record,
    parse_args,
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


def test_extract_dapo_reward_model_answer_and_raw_problem():
    record = {
        "prompt": [{"role": "user", "content": "Old instruction that asks for Answer:"}],
        "raw_problem": "Find 3 + 4.",
        "raw_problem_id": "raw-7",
        "data_source": "DAPO",
        "ability": "MATH",
        "reward_model": {"ground_truth": "7", "style": "rule-lighteval/MATH_v2"},
    }

    example = normalize_dataset_record(record, dataset_id="BytedTsinghua-SIA/DAPO-Math-17k", index=0)

    assert example.id == "raw-7"
    assert example.prompt == "Find 3 + 4."
    assert example.answer == "7"
    assert example.metadata["data_source"] == "DAPO"


def test_extract_olympiadbench_list_answer_metadata():
    record = {
        "id": 1606,
        "question": "How many moves are needed?",
        "final_answer": ["2"],
        "subfield": "Combinatorics",
        "language": "English",
        "answer_type": "Numerical",
        "is_multiple_answer": False,
    }

    example = normalize_dataset_record(record, dataset_id="Hothan/OlympiadBench", index=0)

    assert example.id == "1606"
    assert example.answer == "2"
    assert example.metadata["subfield"] == "Combinatorics"
    assert example.metadata["answer_type"] == "Numerical"


def test_extract_rlvr_verifier_answer():
    record = {
        "id": "rlvr-1",
        "prompt": [{"role": "user", "content": "Compute 5+7."}],
        "verifier": {"type": "math_boxed_v001", "answer": "12"},
    }

    assert extract_prompt(record) == "Compute 5+7."
    assert extract_answer(record) == "12"


def test_numeric_zero_id_is_preserved():
    record = {
        "id": 0,
        "question": "Compute 1+1.",
        "answer": 2,
    }

    example = normalize_dataset_record(record, dataset_id="AMC23", index=0)

    assert example.id == "0"
    assert example.answer == "2"


def test_format_prompt_requires_single_final_boxed_answer():
    prompt = format_math_prompt("Compute 1+1.")

    assert "exactly one \\boxed{...}" in prompt
    assert "Do not write anything after the boxed answer." in prompt
    assert "Compute 1+1." in prompt


def test_paper_math_prompt_template_is_explicitly_selectable():
    prompt = format_math_prompt("Compute 1+1.", template="paper_math")

    assert "step by step" in prompt
    assert "exactly one final answer in \\boxed{...}" in prompt
    assert "nothing after the boxed answer" in prompt


def test_paper_dapo_prompt_template_matches_paper_suffix():
    prompt = format_math_prompt("Compute 1+1.", template="paper_dapo")

    assert prompt.startswith("Compute 1+1.")
    assert "Let's think step by step" in prompt
    assert "within \\boxed{}" in prompt


def test_parse_args_accepts_symbolic_equivalence_engine():
    args = parse_args(
        [
            "--model-name",
            "dummy-model",
            "--output-dir",
            "runs/eval/dummy",
            "--allow-symbolic-equivalence",
            "--symbolic-equivalence-engine",
            "sympy",
        ]
    )

    assert args.allow_symbolic_equivalence is True
    assert args.symbolic_equivalence_engine == "sympy"


def test_parse_args_accepts_extra_eos_tokens():
    args = parse_args(
        [
            "--model-name",
            "dummy-model",
            "--output-dir",
            "runs/eval/dummy",
            "--extra-eos-token",
            "<|im_end|>",
        ]
    )

    assert args.extra_eos_token == ["<|im_end|>"]


def test_parse_args_accepts_num_samples_per_example():
    args = parse_args(
        [
            "--model-name",
            "dummy-model",
            "--output-dir",
            "runs/eval/dummy",
            "--num-samples-per-example",
            "16",
        ]
    )

    assert args.num_samples_per_example == 16


def test_parse_args_accepts_vllm_backend_options():
    args = parse_args(
        [
            "--model-name",
            "dummy-model",
            "--output-dir",
            "runs/eval/dummy",
            "--inference-backend",
            "vllm",
            "--vllm-tensor-parallel-size",
            "2",
            "--vllm-gpu-memory-utilization",
            "0.75",
            "--vllm-max-model-len",
            "4096",
            "--vllm-dtype",
            "bfloat16",
        ]
    )

    assert args.inference_backend == "vllm"
    assert args.vllm_tensor_parallel_size == 2
    assert args.vllm_gpu_memory_utilization == 0.75
    assert args.vllm_max_model_len == 4096
    assert args.vllm_dtype == "bfloat16"


def test_load_local_jsonl_eval_examples(tmp_path):
    data_path = tmp_path / "eval.jsonl"
    data_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "a",
                        "prompt": [{"role": "user", "content": "Compute 1+2."}],
                        "verifier": {"answer": "3"},
                    }
                ),
                json.dumps(
                    {
                        "id": "b",
                        "problem": "Compute 2+2.",
                        "answer": "4",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    examples = load_eval_examples(
        dataset_id="local-smoke",
        dataset_path=str(data_path),
        dataset_format="auto",
        sample_size=2,
        shuffle_buffer_size=1,
    )

    assert [example.id for example in examples] == ["a", "b"]
    assert [example.answer for example in examples] == ["3", "4"]


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
    assert summary["full_credit_accuracy"] == pytest.approx(1 / 3)
    assert summary["reward_mean"] == pytest.approx(1 / 3)
    assert summary["parse_failure_rate"] == pytest.approx(1 / 3)
    assert summary["format_success_rate"] == pytest.approx(2 / 3)
    assert summary["correctness_given_parse"] == 0.5
    assert summary["truncation_rate"] == 1 / 3
    assert summary["reason_counts"] == {"answer_mismatch": 1, "exact_match": 1, "no_boxed_answer": 1}


def test_summarize_eval_rows_keeps_partial_reward_out_of_accuracy():
    rows = [
        {
            "reward": 1.0,
            "reason": "exact_match",
            "parse_failed": False,
            "completion_chars": 10,
            "completion_tokens": 2,
            "truncated": False,
            "metadata": {},
        },
        {
            "reward": 0.1,
            "reason": "format_correct_answer_mismatch",
            "parse_failed": False,
            "completion_chars": 10,
            "completion_tokens": 2,
            "truncated": False,
            "metadata": {},
        },
    ]

    summary = summarize_eval_rows(rows, {})

    assert summary["accuracy"] == 0.5
    assert summary["full_credit_accuracy"] == 0.5
    assert summary["reward_mean"] == pytest.approx(0.55)


def test_summarize_eval_rows_reports_pass_at_n_by_example():
    rows = [
        {
            "id": "a",
            "reward": 0.0,
            "reason": "format_correct_answer_mismatch",
            "parse_failed": False,
            "completion_chars": 10,
            "completion_tokens": 2,
            "truncated": False,
            "metadata": {},
        },
        {
            "id": "a",
            "reward": 1.0,
            "reason": "exact_match",
            "parse_failed": False,
            "completion_chars": 10,
            "completion_tokens": 2,
            "truncated": False,
            "metadata": {},
        },
        {
            "id": "b",
            "reward": 0.0,
            "reason": "no_boxed_answer",
            "parse_failed": True,
            "completion_chars": 10,
            "completion_tokens": 2,
            "truncated": False,
            "metadata": {},
        },
        {
            "id": "b",
            "reward": 0.0,
            "reason": "format_correct_answer_mismatch",
            "parse_failed": False,
            "completion_chars": 10,
            "completion_tokens": 2,
            "truncated": False,
            "metadata": {},
        },
    ]

    summary = summarize_eval_rows(rows, {"num_samples_per_example": 2})

    assert summary["num_examples"] == 2
    assert summary["num_completions"] == 4
    assert summary["accuracy"] == 0.25
    assert summary["pass_at_n"] == 0.5
    assert summary["all_correct_prompt_rate"] == 0.0
    assert summary["all_wrong_prompt_rate"] == 0.5


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
            "dataset_path": None,
            "dataset_format": "auto",
            "file_glob": "**/*.jsonl,**/*.json,**/*.parquet",
            "prompt_template": "boxed",
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
