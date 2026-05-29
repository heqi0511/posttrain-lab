import json
from collections import Counter

from posttrain_lab.data.openr1_level_filter import (
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
