import json
from pathlib import Path

from posttrain_lab.pipelines.e2e_smoke import run_e2e_smoke


def test_e2e_smoke_safe_mode_writes_required_artifacts(tmp_path):
    output_dir = tmp_path / "e2e_smoke"

    result = run_e2e_smoke(
        "configs/e2e/toy_math_posttraining.yaml",
        output_dir=output_dir,
        real_run=False,
    )

    assert result["mode"] == "dry-run"
    assert result["output_dir"] == str(output_dir)

    required = [
        "baseline_eval_report.json",
        "sft_run_card.md",
        "sft_eval_report.json",
        "rlvr_run_card.md",
        "rlvr_eval_report.json",
        "comparison_report.md",
        "sample_generations.jsonl",
        "sample_rollouts.jsonl",
    ]
    for filename in required:
        assert (output_dir / filename).exists(), filename

    report = (output_dir / "comparison_report.md").read_text(encoding="utf-8")
    assert "dry-run" in report
    assert "target accuracy" in report
    assert "format success rate" in report
    assert "parse failure rate" in report
    assert "average output length" in report
    assert "reward mean" in report
    assert "reward std" in report

    baseline = json.loads((output_dir / "baseline_eval_report.json").read_text(encoding="utf-8"))
    assert baseline["metrics"]["exact_match"] == 1.0
    assert baseline["metrics"]["parse_failure_rate"] == 0.0

    staged_sft = [
        json.loads(line)
        for line in (output_dir / "staged_data" / "sft_smoke.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert sum(row["split"] == "train" for row in staged_sft) == 80
    assert sum(row["split"] == "val" for row in staged_sft) == 10
    assert {row["metadata"]["difficulty"] for row in staged_sft} == {"easy", "medium", "hard"}
    assert {row["metadata"]["domain"] for row in staged_sft} >= {
        "arithmetic",
        "fractions",
        "linear_equations",
        "algebra_simplification",
    }

    assert (output_dir / "sample_generations.jsonl").read_text(encoding="utf-8").strip()
    assert (output_dir / "sample_rollouts.jsonl").read_text(encoding="utf-8").strip()
