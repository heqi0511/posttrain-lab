"""Compare base, SFT, and SFT+RLVR eval runs without changing eval artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from posttrain_lab.rewards.math_reward import MathRewardConfig, score_math_boxed_v001


RUN_ORDER = ["base", "sft", "sft_rlvr"]
RUN_LABELS = {
    "base": "Base",
    "sft": "SFT",
    "sft_rlvr": "SFT+RLVR",
}
PARSE_FAILURE_REASONS = {
    "no_boxed_answer",
    "malformed_boxed_answer",
    "empty_boxed_answer",
    "boxed_not_final_only",
    "unclosed_think_block",
}


def compare_run_dirs(
    base_dir,
    sft_dir,
    rlvr_dir,
    output_path,
    failure_output_path,
    *,
    too_long_threshold=512,
    failure_limit=50,
):
    """Write a comparison report and RLVR failure-case JSONL."""

    run_dirs = {
        "base": Path(base_dir),
        "sft": Path(sft_dir),
        "sft_rlvr": Path(rlvr_dir),
    }
    loaded = {name: _load_eval_run(path) for name, path in run_dirs.items()}
    _require_same_eval_suite(loaded)

    failure_rows = _extract_failure_rows(
        loaded["sft_rlvr"]["rows"],
        limit=int(failure_limit),
        too_long_threshold=int(too_long_threshold),
    )
    _write_jsonl(failure_output_path, failure_rows)

    report = _build_report(
        loaded=loaded,
        run_dirs=run_dirs,
        failure_rows=failure_rows,
        failure_output_path=Path(failure_output_path),
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(report, encoding="utf-8")
    return {
        "output_path": str(output_path),
        "failure_output_path": str(failure_output_path),
        "failure_count": len(failure_rows),
        "heldout_conclusion": _heldout_conclusion(loaded),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare base, SFT, and SFT+RLVR eval runs.")
    parser.add_argument("--base", required=True, help="Eval run directory for the base model.")
    parser.add_argument("--sft", required=True, help="Eval run directory for the SFT adapter.")
    parser.add_argument("--rlvr", required=True, help="Eval run directory for the SFT+RLVR adapter.")
    parser.add_argument("--output", required=True, help="Path for comparison_report.md.")
    parser.add_argument("--failure-output", required=True, help="Path for extracted RLVR failures JSONL.")
    parser.add_argument("--failure-limit", type=int, default=50)
    parser.add_argument("--too-long-threshold", type=int, default=512)
    args = parser.parse_args(argv)

    result = compare_run_dirs(
        args.base,
        args.sft,
        args.rlvr,
        args.output,
        args.failure_output,
        too_long_threshold=args.too_long_threshold,
        failure_limit=args.failure_limit,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def _load_eval_run(path):
    path = Path(path)
    metrics_path = path / "metrics.json"
    rows_path = path / "raw_generations.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing eval metrics: {metrics_path}")
    if not rows_path.exists():
        raise FileNotFoundError(f"missing raw generations: {rows_path}")
    return {
        "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
        "rows": _read_jsonl(rows_path),
    }


def _require_same_eval_suite(loaded):
    signatures = {name: _eval_signature(run["rows"]) for name, run in loaded.items()}
    reference = signatures["base"]
    mismatched = [name for name, signature in signatures.items() if signature != reference]
    if mismatched:
        raise ValueError(
            "eval suite mismatch across runs; refusing comparison for: "
            + ", ".join(sorted(mismatched))
        )


def _eval_signature(rows):
    return [
        (
            str(row.get("id")),
            json.dumps(row.get("prompt"), sort_keys=True),
            str(row.get("answer")),
        )
        for row in rows
    ]


def _extract_failure_rows(rows, limit, too_long_threshold):
    failures = []
    for row in rows:
        if not _is_failure(row):
            continue
        generation = str(row.get("generation", ""))
        answer = _unbox_answer(str(row.get("answer", "")))
        reward = score_math_boxed_v001(generation, answer, config=MathRewardConfig())
        failures.append(
            {
                "id": row.get("id"),
                "category": _categorize_failure(row, reward, too_long_threshold),
                "prompt": row.get("prompt"),
                "answer": row.get("answer"),
                "generation": generation,
                "exact_match": row.get("exact_match"),
                "format_success": row.get("format_success"),
                "parsed_answer": reward.normalized_prediction,
                "failure_reason": reward.reason,
                "completion_length": row.get("completion_length"),
            }
        )
        if len(failures) == limit:
            break
    return failures


def _is_failure(row):
    if row.get("exact_match") is False:
        return True
    if row.get("format_success") is False:
        return True
    return False


def _categorize_failure(row, reward, too_long_threshold):
    generation = str(row.get("generation", ""))
    target = _unbox_answer(str(row.get("answer", "")))
    if len(generation) > too_long_threshold or reward.reason == "output_too_long":
        return "too long"
    if reward.reason in {"multiple_boxed_answers", "conflicting_boxed_answers"}:
        return "multiple answers"
    if generation.count(r"\boxed{") > 1:
        return "multiple answers"
    if reward.reason in PARSE_FAILURE_REASONS:
        return "parser failure"
    if row.get("format_success") is False:
        return "format violation"
    if target and target in generation and reward.normalized_prediction != target:
        return "correct reasoning wrong final"
    return "wrong reasoning"


def _build_report(loaded, run_dirs, failure_rows, failure_output_path):
    lines = [
        "# Base / SFT / SFT+RLVR Comparison",
        "",
        "This report compares heldout eval metrics only; it does not use training reward as evidence of improvement.",
        "",
        f"- eval suite consistency: `{_same_eval_suite_label(loaded)}`",
        f"- RLVR failure cases extracted: `{len(failure_rows)}`",
        f"- failure output: `{failure_output_path}`",
        "",
        "## Comparison Table",
        "",
        "| run | target accuracy | format success | parse failure rate | average output length | general regression score |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run_name in RUN_ORDER:
        metrics = loaded[run_name]["metrics"]
        lines.append(
            "| {label} | {target} | {format_success} | {parse_failure} | {avg_len} | {general} |".format(
                label=RUN_LABELS[run_name],
                target=_fmt_metric(metrics.get("exact_match")),
                format_success=_fmt_metric(metrics.get("format_success")),
                parse_failure=_fmt_metric(metrics.get("parse_failure_rate")),
                avg_len=_fmt_metric(metrics.get("completion_length_mean")),
                general=_fmt_metric(_general_regression_score(metrics)),
            )
        )

    lines.extend(
        [
            "",
            "## Heldout Eval Conclusion",
            "",
            _heldout_conclusion(loaded),
            "",
            "## Failure Taxonomy",
            "",
            "| category | count |",
            "| --- | ---: |",
        ]
    )
    counts = _failure_counts(failure_rows)
    for category in [
        "wrong reasoning",
        "correct reasoning wrong final",
        "parser failure",
        "multiple answers",
        "too long",
        "format violation",
    ]:
        lines.append(f"| {category} | {counts.get(category, 0)} |")

    lines.extend(
        [
            "",
            "## Inputs",
            "",
        ]
    )
    for run_name in RUN_ORDER:
        lines.append(f"- {RUN_LABELS[run_name]} eval dir: `{run_dirs[run_name]}`")
    lines.append("")
    lines.append(
        "General regression score is read from eval metrics when present; otherwise it is reported as `n/a` "
        "because the current MVP eval runner has only target metrics."
    )
    lines.append("")
    return "\n".join(lines)


def _heldout_conclusion(loaded):
    sft_score = loaded["sft"]["metrics"].get("exact_match")
    rlvr_score = loaded["sft_rlvr"]["metrics"].get("exact_match")
    if not isinstance(sft_score, (int, float)) or not isinstance(rlvr_score, (int, float)):
        return "RLVR heldout improvement is unavailable because target accuracy is missing."
    delta = rlvr_score - sft_score
    if delta > 0:
        return f"RLVR improved heldout target accuracy over SFT by {delta:.4f}."
    if delta < 0:
        return f"RLVR regressed heldout target accuracy versus SFT by {delta:.4f}."
    return "RLVR did not change heldout target accuracy versus SFT."


def _same_eval_suite_label(loaded):
    _require_same_eval_suite(loaded)
    return "same prompts, answers, and example ids"


def _failure_counts(rows):
    counts = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    return counts


def _general_regression_score(metrics):
    for key in ("general_regression_score", "general_score", "general_exact_match"):
        if key in metrics:
            return metrics[key]
    return None


def _fmt_metric(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _unbox_answer(answer):
    match = re.fullmatch(r"\\boxed\{(.+)\}", answer.strip())
    if match:
        return match.group(1).strip()
    return answer.strip()


def _read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
    return rows


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
