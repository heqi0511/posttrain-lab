import json

from posttrain_lab.data.target_policy_filter import filter_dataset_dirs
from posttrain_lab.data.validate import validate_jsonl


def test_target_policy_filter_preserves_splits_and_rejects_assignments(tmp_path):
    rlvr_in = tmp_path / "rlvr_in"
    sft_in = tmp_path / "sft_in"
    rlvr_in.mkdir()
    sft_in.mkdir()
    for split in ("train", "val", "test"):
        _write_jsonl(
            rlvr_in / f"{split}.jsonl",
            [
                _rlvr(f"keep-{split}", split, "2"),
                _rlvr(f"assignment-{split}", split, "x=2"),
                _rlvr(f"pm-{split}", split, r"\pm\sqrt{3}"),
            ],
        )
        _write_jsonl(
            sft_in / f"{split}.jsonl",
            [
                _sft(f"keep-{split}", split, r"\boxed{2}"),
                _sft(f"assignment-{split}", split, r"\boxed{x=2}"),
                _sft(f"pm-{split}", split, r"\boxed{\pm\sqrt{3}}"),
            ],
        )

    manifest = filter_dataset_dirs(
        input_rlvr_dir=rlvr_in,
        output_rlvr_dir=tmp_path / "rlvr_out",
        input_sft_dir=sft_in,
        output_sft_dir=tmp_path / "sft_out",
        target_policy="single_expression_no_assignment_v1",
    )

    assert manifest["split_membership_preserved"] is True
    assert manifest["input_counts"] == {"train": 3, "val": 3, "test": 3}
    assert manifest["split_counts"] == {"train": 1, "val": 1, "test": 1}
    assert manifest["reject_counts"] == {"assignment_target": 3, "pm_target": 3}
    assert validate_jsonl("rlvr", tmp_path / "rlvr_out" / "train.jsonl") == []
    assert validate_jsonl("sft", tmp_path / "sft_out" / "train.jsonl") == []

    kept = [json.loads(line) for line in (tmp_path / "rlvr_out" / "train.jsonl").read_text().splitlines()]
    assert [row["id"] for row in kept] == ["keep-train"]


def _rlvr(row_id, split, answer):
    return {
        "id": row_id,
        "split": split,
        "prompt": [{"role": "user", "content": "Find the value."}],
        "verifier": {"type": "math_boxed_v001", "answer": answer},
        "metadata": {
            "source": "fixture",
            "domain": "Algebra",
            "difficulty": "fixture",
            "license": "fixture",
        },
    }


def _sft(row_id, split, assistant_content):
    return {
        "id": row_id,
        "split": split,
        "messages": [
            {"role": "user", "content": "Find the value."},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "source": "fixture",
            "domain": "Algebra",
            "difficulty": "fixture",
            "license": "fixture",
        },
    }


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
