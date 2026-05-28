import json

import pytest

from posttrain_lab.data.gsm8k import convert_gsm8k_to_rlvr, parse_gsm8k_final_answer
from posttrain_lab.data.validate import validate_file


def test_parse_gsm8k_final_answer_from_hash_field():
    assert parse_gsm8k_final_answer("reasoning\n#### 72") == "72"
    assert parse_gsm8k_final_answer("reasoning\n#### $1,234.") == "1234"


def test_parse_gsm8k_final_answer_rejects_missing_hash_field():
    with pytest.raises(ValueError, match="missing final ####"):
        parse_gsm8k_final_answer("answer is 72")


def test_convert_gsm8k_train_to_rlvr_jsonl_without_test_split(tmp_path):
    input_path = tmp_path / "gsm8k_sample.jsonl"
    output_path = tmp_path / "gsm8k_train.jsonl"
    summary_path = tmp_path / "summary.json"
    rows = [
        {
            "question": "Jan has 3 apples and buys 4 more. How many apples?",
            "answer": "Jan has 3 + 4 = 7 apples.\n#### 7",
        },
        {
            "question": "A box has 1,200 marbles and loses 200. How many remain?",
            "answer": "1200 - 200 = 1000.\n#### 1,000",
        },
    ]
    input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = convert_gsm8k_to_rlvr(
        output_path,
        input_jsonl=input_path,
        split="train",
        summary_path=summary_path,
    )

    assert summary["num_rows"] == 2
    assert summary["schema_valid"] is True
    assert summary["no_gsm8k_test_examples_used"] is True
    assert json.loads(summary_path.read_text(encoding="utf-8"))["source_split"] == "train"

    report = validate_file(output_path, "rlvr")
    assert report.ok, report.errors

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["id"] == "gsm8k-train-000000"
    assert records[0]["split"] == "train"
    assert records[0]["verifier"] == {"type": "math_boxed_v001", "answer": "7"}
    assert records[1]["verifier"]["answer"] == "1000"
    assert "####" not in records[0]["prompt"][0]["content"]
    assert "Do not include reasoning" in records[0]["prompt"][0]["content"]


def test_convert_gsm8k_refuses_test_split_for_rlvr_train(tmp_path):
    with pytest.raises(ValueError, match="only the official GSM8K train split"):
        convert_gsm8k_to_rlvr(tmp_path / "out.jsonl", input_jsonl=tmp_path / "unused.jsonl", split="test")
