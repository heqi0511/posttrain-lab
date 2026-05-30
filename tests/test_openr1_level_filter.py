import json
from collections import Counter

from posttrain_lab.data.openr1_level_filter import (
    build_openr1_level_filtered_dataset,
    collect_filtered_prompts,
    extract_openr1_level_record,
)
from posttrain_lab.data.validate import validate_jsonl


def test_extract_openr1_level_record_accepts_level_strings():
    record = {
        "uuid": "abc",
        "problem": "Find x.",
        "answer": r"\boxed{2}",
        "problem_type": "Algebra",
        "level": "Level 3",
    }

    extracted = extract_openr1_level_record(record)

    assert extracted["source_id"] == "abc"
    assert extracted["domain"] == "Algebra"
    assert extracted["level"] == 3


def test_collect_filtered_prompts_keeps_only_selected_levels_and_domains():
    records = [
        {
            "uuid": "keep-algebra",
            "problem": "Solve $x+1=3$.",
            "answer": r"\boxed{2}",
            "problem_type": "Algebra",
            "level": 2,
        },
        {
            "uuid": "keep-number-theory",
            "problem": "Find the remainder when $17$ is divided by $5$.",
            "answer": r"\boxed{2}",
            "problem_type": "Number Theory",
            "level": "3",
        },
        {
            "uuid": "reject-level",
            "problem": "Compute $2+2$.",
            "answer": r"\boxed{4}",
            "problem_type": "Algebra",
            "level": 4,
        },
        {
            "uuid": "reject-domain",
            "problem": "Find the area of a square of side $3$.",
            "answer": r"\boxed{9}",
            "problem_type": "Geometry",
            "level": 2,
        },
        {
            "uuid": "reject-missing-level",
            "problem": "Solve $y=1$.",
            "answer": r"\boxed{1}",
            "problem_type": "Algebra",
        },
    ]
    rejects = Counter()

    prompts = collect_filtered_prompts(
        records,
        needed=10,
        levels=(2, 3),
        domains=("Algebra", "Number Theory"),
        require_parser=False,
        reject_counts=rejects,
    )

    assert [prompt.source_id for prompt in prompts] == ["keep-algebra", "keep-number-theory"]
    assert prompts[0].answer == "2"
    assert rejects["level_not_selected"] == 1
    assert rejects["domain_not_selected"] == 1
    assert rejects["missing_level"] == 1


def test_collect_filtered_prompts_can_filter_sources_without_level():
    records = [
        {
            "uuid": "keep-cn",
            "problem": "Solve $x+1=3$.",
            "answer": r"\boxed{2}",
            "problem_type": "Algebra",
            "source": "cn_k12",
        },
        {
            "uuid": "reject-source",
            "problem": "Solve $x+2=4$.",
            "answer": r"\boxed{2}",
            "problem_type": "Algebra",
            "source": "other",
        },
    ]
    rejects = Counter()

    prompts = collect_filtered_prompts(
        records,
        needed=10,
        levels=(),
        domains=("Algebra", "Number Theory"),
        sources=("cn_k12", "cn_contest", "amc_aime"),
        require_parser=False,
        reject_counts=rejects,
    )

    assert [prompt.source_id for prompt in prompts] == ["keep-cn"]
    assert prompts[0].source == "cn_k12"
    assert prompts[0].level is None
    assert rejects["source_not_selected"] == 1


def test_level_filtered_rlvr_records_validate(tmp_path):
    records = [
        {
            "uuid": "a",
            "problem": "Solve $x+1=3$.",
            "answer": r"\boxed{2}",
            "problem_type": "Algebra",
            "level": 2,
        }
    ]
    prompts = collect_filtered_prompts(records, needed=1, require_parser=False)
    row = {
        "id": "openr1-l23-algebra-train-000000",
        "split": "train",
        "prompt": [
            {
                "role": "user",
                "content": prompts[0].problem + "\n\nReturn only the final answer in exactly one \\boxed{...}.",
            }
        ],
        "verifier": {"type": "math_boxed_v001", "answer": prompts[0].answer},
        "metadata": {
            "source": "open-r1/OpenR1-Math-220k",
            "domain": "Algebra",
            "difficulty": "level_2",
            "license": "source-dataset-card",
        },
    }
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    assert validate_jsonl("rlvr", path) == []


def test_build_level_filtered_dataset_writes_rlvr_and_sft(tmp_path):
    records = [
        {
            "uuid": "a",
            "problem": "Solve $x+1=3$.",
            "answer": r"\boxed{2}",
            "problem_type": "Algebra",
            "source": "cn_k12",
        },
        {
            "uuid": "b",
            "problem": "Find the remainder when $17$ is divided by $5$.",
            "answer": r"\boxed{2}",
            "problem_type": "Number Theory",
            "source": "amc_aime",
        },
    ]

    manifest = build_openr1_level_filtered_dataset(
        output_dir=tmp_path / "rlvr",
        sft_output_dir=tmp_path / "sft",
        levels=(),
        sources=("cn_k12", "amc_aime"),
        train_count=1,
        val_count=1,
        test_count=0,
        require_parser=False,
        source_records=records,
    )

    assert manifest["split_counts"] == {"train": 1, "val": 1, "test": 0}
    assert validate_jsonl("rlvr", tmp_path / "rlvr" / "train.jsonl") == []
    assert validate_jsonl("sft", tmp_path / "sft" / "train.jsonl") == []
