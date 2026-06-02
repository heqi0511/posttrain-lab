"""Convert DAPO-Math raw data into strict RLVR JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from posttrain_lab.data.validate import validate_file


DEFAULT_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. The last line of your "
    "response must contain exactly one final answer in \\boxed{{...}}, with "
    "nothing after the boxed answer.\n\n"
    "{question}"
)


def convert_dapo_to_rlvr(
    output_path,
    *,
    input_path=None,
    dataset_name="BytedTsinghua-SIA/DAPO-Math-17k",
    split="train",
    max_examples: Optional[int] = None,
    prompt_template=DEFAULT_PROMPT_TEMPLATE,
    summary_path=None,
):
    """Write DAPO train examples as RLVR JSONL and return a conversion summary."""

    if split != "train":
        raise ValueError("only the DAPO train split may be converted for RLVR training")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = []
    with output_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(_load_rows(input_path=input_path, dataset_name=dataset_name, split=split)):
            if max_examples is not None and written >= int(max_examples):
                break
            problem = _extract_problem(row)
            answer = _extract_answer(row)
            if not problem or not answer:
                skipped.append({"index": index, "reason": "missing_problem_or_answer"})
                continue

            record = {
                "id": _stable_id(row, written),
                "split": "train",
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt_template.format(question=problem),
                    }
                ],
                "verifier": {"type": "math_boxed_v001", "answer": answer},
                "metadata": {
                    "source": _source(row, dataset_name),
                    "domain": str(row.get("ability") or "math"),
                    "difficulty": "mixed",
                    "license": "source-dataset-card",
                },
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            written += 1

    report = validate_file(output_path, "rlvr")
    summary = {
        "dataset_name": dataset_name,
        "source_split": split,
        "input_path": str(input_path) if input_path else None,
        "output_path": str(output_path),
        "num_rows": written,
        "skipped_count": len(skipped),
        "skipped_examples": skipped[:20],
        "schema_valid": report.ok,
        "schema_error_count": len(report.errors),
    }
    if summary_path is not None:
        _write_text(Path(summary_path), json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if not report.ok:
        raise ValueError("\n".join(str(error) for error in report.errors))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert DAPO-Math train data into RLVR JSONL.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--input-path", default=None, help="Local JSONL/JSON/Parquet path, if already downloaded.")
    parser.add_argument("--dataset-name", default="BytedTsinghua-SIA/DAPO-Math-17k")
    parser.add_argument("--split", default="train", choices=["train"])
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args(argv)

    summary = convert_dapo_to_rlvr(
        args.output,
        input_path=args.input_path,
        dataset_name=args.dataset_name,
        split=args.split,
        max_examples=args.max_examples,
        summary_path=args.summary,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


def _load_rows(input_path, dataset_name, split) -> Iterable[Dict[str, Any]]:
    if input_path:
        path = Path(input_path)
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            yield from _load_jsonl(path)
            return
        if suffix == ".json":
            yield from _load_json(path)
            return
        if suffix == ".parquet":
            yield from _load_dataset_rows("parquet", data_files=str(path), split="train")
            return
        raise ValueError(f"unsupported DAPO input file format: {path}")

    yield from _load_dataset_rows(dataset_name, split=split)


def _load_dataset_rows(path, *, split, data_files=None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install the optional 'datasets' package or pass a local JSONL/JSON file") from exc

    if data_files is None:
        dataset = load_dataset(path, split=split)
    else:
        dataset = load_dataset(path, data_files=data_files, split=split)
    for row in dataset:
        yield dict(row)


def _load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield row


def _load_json(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        payload = payload["data"]
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected JSON list or object with data list")
    for index, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{index}: expected JSON object")
        yield row


def _extract_problem(row):
    value = row.get("raw_problem")
    if isinstance(value, str) and value.strip():
        return value.strip()

    prompt = row.get("prompt")
    if isinstance(prompt, list):
        parts = [
            str(message.get("content", "")).strip()
            for message in prompt
            if isinstance(message, dict) and message.get("role") == "user" and str(message.get("content", "")).strip()
        ]
        if parts:
            return "\n".join(parts)
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    return ""


def _extract_answer(row):
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict):
        answer = reward_model.get("ground_truth")
        if answer is not None and str(answer).strip():
            return str(answer).strip()
    for key in ("answer", "final_answer", "target"):
        answer = row.get(key)
        if answer is not None and str(answer).strip():
            return str(answer).strip()
    return ""


def _stable_id(row, index):
    for key in ("raw_problem_id", "id", "uuid", "problem_id"):
        value = row.get(key)
        if value is not None and value != "":
            return str(value)
    extra_info = row.get("extra_info")
    if isinstance(extra_info, dict):
        value = extra_info.get("index")
        if value is not None and value != "":
            return str(value)
    return f"dapo-math-train-{index:06d}"


def _source(row, dataset_name):
    data_source = row.get("data_source")
    if data_source:
        return f"{dataset_name}:{data_source}"
    return dataset_name


def _write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
