"""Classify parse failures from rollout audit completions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

from posttrain_lab.rewards.math_reward import normalize_math_answer, score_math_boxed_v001


CATEGORIES = (
    "no_boxed_answer",
    "truncated_before_final",
    "final_answer_unboxed",
    "malformed_boxed",
    "multiple_conflicting_boxed_answers",
    "non_numeric_boxed_answer",
    "parser_too_strict",
    "thinking_mode_interference",
    "other",
)

PARSE_FAILURE_REASONS = {
    "no_boxed_answer",
    "malformed_boxed_answer",
    "empty_boxed_answer",
    "conflicting_boxed_answers",
    "multiple_boxed_answers",
    "boxed_not_final_only",
    "unclosed_think_block",
    "output_too_long",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Classify rollout parse failures without changing reward semantics.")
    parser.add_argument("--input", default=None, help="Rollout JSONL file. Defaults to latest audit completions file.")
    parser.add_argument("--config", default=None, help="Optional resolved_config.yaml with rollout.max_new_tokens.")
    parser.add_argument("--output-dir", default="data/reports/parse_failure_taxonomy")
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--chars-per-token", type=float, default=3.2)
    args = parser.parse_args(argv)

    input_path = Path(args.input) if args.input else find_latest_audit_completions()
    if input_path is None:
        raise FileNotFoundError("no audit completions file found under runs/**/sample_rollouts_for_review.jsonl")
    config_path = Path(args.config) if args.config else _default_config_path(input_path)
    max_new_tokens = _read_max_new_tokens(config_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    classified = [
        classify_row(
            row,
            row_index=index,
            max_new_tokens=max_new_tokens,
            chars_per_token=args.chars_per_token,
        )
        for index, row in enumerate(rows, start=1)
        if _is_parse_failure(row)
    ]
    summary = build_summary(
        rows=rows,
        classified=classified,
        input_path=input_path,
        config_path=config_path,
        max_new_tokens=max_new_tokens,
        chars_per_token=args.chars_per_token,
    )

    _write_json(output_dir / "parse_failure_summary.json", summary)
    _write_csv(output_dir / "parse_failure_by_completion.csv", classified)
    _write_examples_md(output_dir / "parse_failure_examples.md", summary, classified, args.max_examples)

    print(json.dumps(summary, sort_keys=True))
    return 0


def find_latest_audit_completions(search_root="runs"):
    root = Path(search_root)
    candidates = list(root.glob("**/sample_rollouts_for_review.jsonl"))
    if not candidates:
        candidates = list(root.glob("**/sample_rollouts.jsonl"))
    if not candidates:
        return None
    return max(candidates, key=_audit_sort_key)


def _audit_sort_key(path):
    summary_path = path.parent / "rollout_audit_summary.json"
    completed = False
    audited_prompt_count = 0
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
        completed = bool(summary.get("completed"))
        audited_prompt_count = int(summary.get("audited_prompt_count") or 0)
    return (completed, audited_prompt_count, path.stat().st_mtime)


def classify_row(row, row_index, max_new_tokens, chars_per_token=3.2):
    completion = str(row.get("completion") or row.get("generation") or "")
    answer = str(row.get("answer") or "")
    reward = score_math_boxed_v001(completion, answer)
    reason = str(row.get("failure_reason") or reward.reason or "")
    visible = _strip_closed_think_blocks(completion)
    boxes, malformed = _extract_boxed_payloads(visible)
    normalized_boxes = [_safe_normalize(value) for value in boxes]
    truncation = _estimate_truncation(row, completion, max_new_tokens, chars_per_token)

    category = _assign_category(
        completion=completion,
        visible=visible,
        answer=answer,
        reason=reason,
        boxes=boxes,
        normalized_boxes=normalized_boxes,
        malformed=malformed,
        truncation=truncation,
    )

    return {
        "row_index": row_index,
        "id": row.get("id", ""),
        "prompt_index": row.get("prompt_index", ""),
        "sample_index": row.get("sample_index", ""),
        "category": category,
        "failure_reason": reason,
        "reward": row.get("reward", reward.score),
        "answer": answer,
        "parsed_answer": row.get("parsed_answer") or reward.normalized_prediction or "",
        "boxed_answer_count": len(boxes),
        "boxed_answers": " | ".join(boxes),
        "normalized_boxed_answers": " | ".join(normalized_boxes),
        "completion_length": row.get("completion_length", len(completion)),
        "estimated_token_count": truncation["estimated_token_count"],
        "reached_max_new_tokens": truncation["reached_max_new_tokens"],
        "truncation_estimate_source": truncation["source"],
        "completion_excerpt": _excerpt(completion, 500),
    }


def build_summary(rows, classified, input_path, config_path, max_new_tokens, chars_per_token):
    category_counts = Counter(row["category"] for row in classified)
    reason_counts = Counter(row["failure_reason"] for row in classified)
    total_rows = len(rows)
    parse_failure_rows = len(classified)
    recommendations = _recommend(category_counts, parse_failure_rows)

    return {
        "input_path": str(input_path),
        "config_path": str(config_path) if config_path else None,
        "total_completions_in_file": total_rows,
        "parse_failure_count": parse_failure_rows,
        "parse_failure_rate_in_file": _safe_ratio(parse_failure_rows, total_rows),
        "max_new_tokens": max_new_tokens,
        "chars_per_token_estimate": chars_per_token,
        "category_counts": dict(category_counts),
        "category_fractions_of_parse_failures": {
            name: _safe_ratio(category_counts[name], parse_failure_rows) for name in CATEGORIES
        },
        "failure_reason_counts": dict(reason_counts),
        "recommendations": recommendations,
        "no_training_executed": True,
        "notes": [
            "This diagnostic does not change reward semantics or parser behavior.",
            "Truncation is exact only when token-count fields exist; otherwise it is estimated from completion length.",
            "The current GSM8K scout saved review rollouts for the first 20 prompts, not all 800 completions.",
        ],
    }


def _assign_category(completion, visible, answer, reason, boxes, normalized_boxes, malformed, truncation):
    if reason == "unclosed_think_block" or _has_unclosed_think_block(completion):
        return "thinking_mode_interference"
    if "<think" in completion.lower() and "</think>" not in completion.lower():
        return "thinking_mode_interference"

    if len(boxes) > 1 and len(set(normalized_boxes)) > 1:
        return "multiple_conflicting_boxed_answers"
    if reason == "conflicting_boxed_answers":
        return "multiple_conflicting_boxed_answers"

    if truncation["reached_max_new_tokens"] and _looks_incomplete(visible):
        return "truncated_before_final"

    if malformed or reason == "malformed_boxed_answer":
        if truncation["reached_max_new_tokens"]:
            return "truncated_before_final"
        return "malformed_boxed"

    if boxes and any(_is_non_numeric(value) for value in normalized_boxes):
        return "non_numeric_boxed_answer"

    if reason == "boxed_not_final_only":
        if _single_correct_box_with_suffix(visible, answer, normalized_boxes):
            return "parser_too_strict"
        return "other"

    if not boxes or reason == "no_boxed_answer":
        if truncation["reached_max_new_tokens"]:
            return "truncated_before_final"
        if _has_unboxed_final_answer(visible):
            return "final_answer_unboxed"
        return "no_boxed_answer"

    return "other"


def _recommend(category_counts, total_parse_failures):
    if total_parse_failures == 0:
        return ["run GRPO smoke"]

    recommendations = []
    truncated = category_counts["truncated_before_final"] / total_parse_failures
    no_boxed = category_counts["no_boxed_answer"] / total_parse_failures
    unboxed = category_counts["final_answer_unboxed"] / total_parse_failures
    strict = category_counts["parser_too_strict"] / total_parse_failures
    thinking = category_counts["thinking_mode_interference"] / total_parse_failures

    if truncated >= 0.25:
        recommendations.append("increase max_new_tokens")
    if no_boxed + unboxed >= 0.25:
        recommendations.append("strengthen prompt")
    if strict >= 0.10:
        recommendations.append("fix parser normalization")
    if thinking >= 0.10:
        recommendations.append("disable thinking mode or allocate a larger thinking budget before scoring")
    if no_boxed + unboxed + strict >= 0.30:
        recommendations.append("run format SFT warmup")
    if not recommendations:
        recommendations.append("run GRPO smoke")
    elif "run GRPO smoke" not in recommendations:
        recommendations.append("do not run GRPO smoke until parse failures are reduced")
    return recommendations


def _is_parse_failure(row):
    if row.get("parse_failure") is True:
        return True
    reason = row.get("failure_reason")
    return reason in PARSE_FAILURE_REASONS


def _default_config_path(input_path):
    candidate = input_path.parent / "resolved_config.yaml"
    return candidate if candidate.exists() else None


def _read_max_new_tokens(path):
    if not path or not Path(path).exists():
        return None
    text = Path(path).read_text(encoding="utf-8")
    match = re.search(r"(?m)^\s+max_new_tokens:\s*(\d+)\s*$", text)
    if match:
        return int(match.group(1))
    match = re.search(r"(?m)^max_new_tokens:\s*(\d+)\s*$", text)
    return int(match.group(1)) if match else None


def _estimate_truncation(row, completion, max_new_tokens, chars_per_token):
    for key in ("completion_token_count", "generated_token_count", "num_generated_tokens"):
        if key in row and row[key] is not None:
            token_count = int(row[key])
            return {
                "estimated_token_count": token_count,
                "reached_max_new_tokens": bool(max_new_tokens and token_count >= max_new_tokens),
                "source": key,
            }

    if max_new_tokens is None:
        return {
            "estimated_token_count": "",
            "reached_max_new_tokens": False,
            "source": "unavailable",
        }

    estimated = int(math.ceil(len(completion) / chars_per_token))
    return {
        "estimated_token_count": estimated,
        "reached_max_new_tokens": estimated >= int(max_new_tokens * 0.95),
        "source": "char_length_heuristic",
    }


def _looks_incomplete(text):
    tail = text.rstrip()[-80:]
    if re.search(r"\\boxed\s*\{[^}]*$", tail):
        return True
    if tail.endswith(("$", "$$", "\\", "=", "+", "-", "*", "/", "and", "is", "the", "answer")):
        return True
    if text.count("{") > text.count("}"):
        return True
    if text.count("$$") % 2 == 1:
        return True
    return False


def _has_unboxed_final_answer(text):
    tail = text.strip()[-240:].lower()
    if r"\boxed" in tail:
        return False
    patterns = (
        r"(?:final answer|answer|therefore|thus|so)[^0-9\-]*-?\$?\d[\d,]*(?:\.\d+)?(?:/\d+)?\s*[.。!！]?$",
        r"(?:=|equals)\s*-?\$?\d[\d,]*(?:\.\d+)?(?:/\d+)?\s*[.。!！]?$",
    )
    return any(re.search(pattern, tail) for pattern in patterns)


def _single_correct_box_with_suffix(text, answer, normalized_boxes):
    if len(normalized_boxes) != 1:
        return False
    if normalized_boxes[0] != _safe_normalize(answer):
        return False
    match = re.search(r"\\boxed\s*\{", text)
    if not match:
        return False
    close_index = _boxed_close_index(text, match.start())
    if close_index is None:
        return False
    suffix = text[close_index + 1 :].strip()
    return bool(suffix and re.search(r"[A-Za-z]", suffix))


def _extract_boxed_payloads(text):
    answers = []
    malformed = False
    start = 0
    while True:
        boxed_index = text.find(r"\boxed", start)
        if boxed_index < 0:
            break
        group_start = boxed_index + len(r"\boxed")
        while group_start < len(text) and text[group_start].isspace():
            group_start += 1
        if group_start >= len(text) or text[group_start] != "{":
            malformed = True
            break
        close_index = _boxed_close_index(text, boxed_index)
        if close_index is None:
            malformed = True
            break
        answers.append(text[group_start + 1 : close_index])
        start = close_index + 1
    return answers, malformed


def _boxed_close_index(text, boxed_index):
    group_start = boxed_index + len(r"\boxed")
    while group_start < len(text) and text[group_start].isspace():
        group_start += 1
    if group_start >= len(text) or text[group_start] != "{":
        return None
    depth = 1
    index = group_start + 1
    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def _strip_closed_think_blocks(text):
    return re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)


def _has_unclosed_think_block(text):
    opens = len(re.findall(r"<think\b[^>]*>", text, flags=re.IGNORECASE))
    closes = len(re.findall(r"</think>", text, flags=re.IGNORECASE))
    return opens > closes


def _safe_normalize(value):
    try:
        return normalize_math_answer(value)
    except Exception:  # pragma: no cover - defensive diagnostics only
        return str(value).strip()


def _is_non_numeric(value):
    value = value.strip()
    if not value:
        return True
    return re.fullmatch(r"-?(?:\d+(?:\.\d+)?|\d+/\d+)", value) is None


def _safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _excerpt(text, limit):
    text = str(text).replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path, rows):
    fieldnames = [
        "row_index",
        "id",
        "prompt_index",
        "sample_index",
        "category",
        "failure_reason",
        "reward",
        "answer",
        "parsed_answer",
        "boxed_answer_count",
        "boxed_answers",
        "normalized_boxed_answers",
        "completion_length",
        "estimated_token_count",
        "reached_max_new_tokens",
        "truncation_estimate_source",
        "completion_excerpt",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_examples_md(path, summary, rows, max_examples):
    grouped = {category: [] for category in CATEGORIES}
    for row in rows:
        grouped[row["category"]].append(row)

    lines = [
        "# Parse Failure Examples",
        "",
        f"Input: `{summary['input_path']}`",
        f"Parse failures in file: `{summary['parse_failure_count']}` / `{summary['total_completions_in_file']}`",
        "",
        "## Recommendations",
        "",
    ]
    for recommendation in summary["recommendations"]:
        lines.append(f"- {recommendation}")
    lines.extend(["", "## Category Summary", ""])
    for category in CATEGORIES:
        count = summary["category_counts"].get(category, 0)
        fraction = summary["category_fractions_of_parse_failures"].get(category, 0.0)
        lines.append(f"- `{category}`: {count} ({fraction:.2%} of parse failures)")

    for category in CATEGORIES:
        examples = grouped[category]
        if not examples:
            continue
        lines.extend(["", f"## {category}", ""])
        for example in examples[:max_examples]:
            lines.extend(
                [
                    f"### {example['id']} sample {example['sample_index']}",
                    "",
                    f"- failure_reason: `{example['failure_reason']}`",
                    f"- answer: `{example['answer']}`",
                    f"- boxed_answers: `{example['boxed_answers']}`",
                    f"- estimated_token_count: `{example['estimated_token_count']}`",
                    f"- reached_max_new_tokens: `{example['reached_max_new_tokens']}`",
                    "",
                    "```text",
                    example["completion_excerpt"],
                    "```",
                    "",
                ]
            )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
