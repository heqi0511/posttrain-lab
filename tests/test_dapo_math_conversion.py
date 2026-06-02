import json

from posttrain_lab.data.dapo_math import convert_dapo_to_rlvr
from posttrain_lab.data.validate import validate_file


def test_convert_dapo_jsonl_to_rlvr(tmp_path):
    input_path = tmp_path / "dapo.jsonl"
    output_path = tmp_path / "train.jsonl"
    summary_path = tmp_path / "summary.json"
    rows = [
        {
            "data_source": "MATH##Aops",
            "raw_problem": "Compute 19 + 23.",
            "raw_problem_id": "problem-1",
            "ability": "MATH",
            "reward_model": {"ground_truth": "42", "style": "rule-lighteval/MATH_v2"},
        },
        {
            "prompt": [{"role": "user", "content": "Compute 5 * 6."}],
            "extra_info": {"index": "problem-2"},
            "reward_model": {"ground_truth": "30"},
        },
    ]
    input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = convert_dapo_to_rlvr(
        output_path,
        input_path=input_path,
        summary_path=summary_path,
    )

    assert summary["num_rows"] == 2
    assert summary["schema_valid"] is True
    assert validate_file(output_path, "rlvr").ok

    staged = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert staged[0]["id"] == "dapo-math-train-000000-problem-1"
    assert staged[0]["verifier"]["answer"] == "42"
    assert "exactly one final answer in \\boxed{...}" in staged[0]["prompt"][0]["content"]
    assert staged[1]["id"] == "dapo-math-train-000001-problem-2"
    assert staged[1]["metadata"]["domain"] == "math"
