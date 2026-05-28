"""Convert GSM8K train split into strict RLVR JSONL."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from posttrain_lab.data.validate import validate_file


DEFAULT_PROMPT_TEMPLATE = (
    "Solve the problem. Return only the final answer in exactly one "
    "\\boxed{{...}}. Do not include reasoning or commas in numbers.\n\n"
    "Problem: {question}"
)


def parse_gsm8k_final_answer(answer_text):
    """Extract and normalize the final GSM8K answer after the last #### marker."""

    if "####" not in answer_text:
        raise ValueError("GSM8K answer is missing final #### marker")
    raw_answer = answer_text.rsplit("####", 1)[1].strip()
    if not raw_answer:
        raise ValueError("GSM8K answer has empty final #### field")

    value = raw_answer.splitlines()[0].strip()
    value = value.strip("$")
    value = value.replace(",", "")
    value = re.sub(r"\s+", "", value)
    if value.startswith("="):
        value = value[1:]
    if value.endswith(".") and not re.search(r"\.\d+$", value):
        value = value[:-1]
    if not value:
        raise ValueError("GSM8K final answer normalized to empty string")
    return value


def convert_gsm8k_to_rlvr(
    output_path,
    *,
    input_jsonl=None,
    dataset_name="openai/gsm8k",
    dataset_config="main",
    split="train",
    max_examples=None,
    prompt_template=DEFAULT_PROMPT_TEMPLATE,
    summary_path=None,
):
    """Write GSM8K train examples as RLVR JSONL and return a conversion summary."""

    if split != "train":
        raise ValueError("only the official GSM8K train split may be converted for RLVR training/audit")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(
        input_jsonl=input_jsonl,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
    )

    written = 0
    parse_errors = []
    with output_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            if max_examples is not None and written >= int(max_examples):
                break
            try:
                answer = parse_gsm8k_final_answer(row["answer"])
            except ValueError as exc:
                parse_errors.append({"index": index, "error": str(exc)})
                continue

            record = {
                "id": f"gsm8k-train-{written:06d}",
                "split": "train",
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt_template.format(question=row["question"].strip()),
                    }
                ],
                "verifier": {"type": "math_boxed_v001", "answer": answer},
                "metadata": {
                    "source": f"gsm8k-{dataset_config}-train",
                    "domain": "math_word_problem",
                    "difficulty": "grade_school",
                    "license": "MIT",
                },
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            written += 1

    report = validate_file(output_path, "rlvr")
    summary = {
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "source_split": split,
        "output_path": str(output_path),
        "num_rows": written,
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors[:20],
        "schema_valid": report.ok,
        "schema_error_count": len(report.errors),
        "no_gsm8k_test_examples_used": split == "train",
    }
    if summary_path is not None:
        _write_text(Path(summary_path), json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if not report.ok:
        raise ValueError("\n".join(str(error) for error in report.errors))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert official GSM8K train data into RLVR JSONL.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--input-jsonl", default=None, help="Optional local GSM8K-style JSONL for tests/offline use.")
    parser.add_argument("--dataset-name", default="openai/gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--split", default="train", choices=["train"])
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args(argv)

    summary = convert_gsm8k_to_rlvr(
        args.output,
        input_jsonl=args.input_jsonl,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        max_examples=args.max_examples,
        summary_path=args.summary,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


def _load_rows(input_jsonl, dataset_name, dataset_config, split):
    if input_jsonl:
        with Path(input_jsonl).open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                if raw_line.strip():
                    row = json.loads(raw_line)
                    yield {"question": row["question"], "answer": row["answer"]}
        return

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install the optional 'datasets' package or pass --input-jsonl") from exc

    try:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
    except Exception:
        if dataset_name != "openai/gsm8k":
            raise
        dataset = load_dataset("gsm8k", dataset_config, split=split)
    for row in dataset:
        yield {"question": row["question"], "answer": row["answer"]}


def _write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
