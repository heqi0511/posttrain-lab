"""Pilot eval for math datasets with boxed-answer scoring."""

from __future__ import annotations

import argparse
import ast
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from posttrain_lab.rewards.math_reward import MathRewardConfig, score_math_boxed_by_version


PARSE_FAILURE_REASONS = {
    "output_too_long",
    "answer_too_long",
    "unclosed_think_block",
    "malformed_boxed_answer",
    "no_boxed_answer",
    "multiple_boxed_answers",
    "conflicting_boxed_answers",
    "boxed_not_final_only",
    "empty_boxed_answer",
}


@dataclass(frozen=True)
class EvalExample:
    """Normalized math eval example."""

    id: str
    prompt: str
    answer: str
    metadata: Dict[str, Any]


def main(argv=None):
    args = parse_args(argv)
    summary = run_math_dataset_eval(vars(args))
    print(json.dumps(summary, sort_keys=True))
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a causal LM on a sampled math dataset.")
    parser.add_argument("--dataset-id", default="local")
    parser.add_argument("--dataset-path")
    parser.add_argument("--dataset-format", choices=["auto", "jsonl", "json", "parquet"], default="auto")
    parser.add_argument("--file-glob", default="**/*.jsonl,**/*.json,**/*.parquet")
    parser.add_argument("--dataset-config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--shuffle-buffer-size", type=int, default=5000)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--enable-thinking", choices=["true", "false", "auto"], default="false")
    parser.add_argument("--prompt-template", choices=["boxed", "paper_math"], default="boxed")
    parser.add_argument("--reward-version", default="math_boxed_v001")
    parser.add_argument("--allow-symbolic-equivalence", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-sample-count", type=int, default=20)
    return parser.parse_args(argv)


def run_math_dataset_eval(config: Dict[str, Any]):
    """Run a sampled math eval and write raw generations plus aggregate metrics."""

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.get("dry_run") and config["dataset_id"] == "dummy":
        examples = [EvalExample(id="dry-000", prompt="Compute 0.", answer="0", metadata={})]
    else:
        examples = load_eval_examples(
            dataset_id=config["dataset_id"],
            dataset_path=config.get("dataset_path"),
            dataset_format=config.get("dataset_format") or "auto",
            file_glob=config.get("file_glob") or "**/*.jsonl,**/*.json,**/*.parquet",
            dataset_config=config.get("dataset_config") or "default",
            split=config.get("split") or "train",
            sample_size=int(config.get("sample_size") or 0),
            seed=int(config.get("seed") or 0),
            shuffle_buffer_size=int(config.get("shuffle_buffer_size") or 0),
        )
    if not examples:
        raise ValueError("no eval examples were loaded")

    generator = DryRunMathGenerator() if config.get("dry_run") else HFBatchedMathGenerator(config)
    reward_config = MathRewardConfig(
        allow_symbolic_equivalence=bool(config.get("allow_symbolic_equivalence", False))
    )
    rows = []

    for batch in _chunks(examples, int(config.get("batch_size") or 1)):
        prompts = [
            format_math_prompt(example.prompt, template=config.get("prompt_template") or "boxed")
            for example in batch
        ]
        generations = generator.generate_batch(prompts)
        for example, completion in zip(batch, generations):
            score = score_math_boxed_by_version(
                completion,
                example.answer,
                reward_version=str(config.get("reward_version") or "math_boxed_v001"),
                config=reward_config,
            )
            completion_tokens = generator.count_completion_tokens(completion)
            rows.append(
                {
                    "id": example.id,
                    "dataset_id": config["dataset_id"],
                    "dataset_config": config.get("dataset_config") or "default",
                    "split": config.get("split") or "train",
                    "prompt": example.prompt,
                    "answer": example.answer,
                    "completion": completion,
                    "reward": score.score,
                    "reason": score.reason,
                    "parsed_answer": score.extracted_answer,
                    "normalized_prediction": score.normalized_prediction,
                    "normalized_answer": score.normalized_answer,
                    "parse_failed": score.reason in PARSE_FAILURE_REASONS,
                    "completion_chars": len(completion),
                    "completion_tokens": completion_tokens,
                    "truncated": completion_tokens >= int(config.get("max_new_tokens") or 0)
                    if completion_tokens is not None
                    else None,
                    "metadata": example.metadata,
                }
            )

    summary = summarize_eval_rows(rows, config)
    _write_json(output_dir / "resolved_config.json", config)
    _write_json(output_dir / "eval_summary.json", summary)
    _write_jsonl(output_dir / "raw_generations.jsonl", rows)
    _write_jsonl(output_dir / "sample_generations.jsonl", select_review_samples(rows, config))
    _write_report(output_dir / "eval_report.md", summary, config)
    return summary


def load_eval_examples(
    dataset_id: str,
    dataset_path: Optional[str] = None,
    dataset_format: str = "auto",
    file_glob: str = "**/*.jsonl,**/*.json,**/*.parquet",
    dataset_config: str = "default",
    split: str = "train",
    sample_size: int = 0,
    seed: int = 0,
    shuffle_buffer_size: int = 0,
) -> List[EvalExample]:
    """Load a deterministic sample from a local path or Hugging Face dataset."""

    if dataset_path:
        records = list(
            iter_local_dataset_records(
                Path(dataset_path),
                dataset_format=dataset_format,
                file_glob=file_glob,
            )
        )
        if shuffle_buffer_size > 1:
            random.Random(seed).shuffle(records)
        if sample_size > 0:
            records = records[:sample_size]
        return [
            normalize_dataset_record(record, dataset_id=dataset_id, index=index)
            for index, record in enumerate(records)
        ]

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for math dataset evals") from exc

    dataset = load_dataset(dataset_id, dataset_config, split=split, streaming=True)
    if shuffle_buffer_size > 1:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
    if sample_size > 0:
        dataset = dataset.take(sample_size)

    examples = []
    for index, record in enumerate(dataset):
        examples.append(normalize_dataset_record(record, dataset_id=dataset_id, index=index))
    return examples


def iter_local_dataset_records(
    dataset_path: Path,
    *,
    dataset_format: str = "auto",
    file_glob: str = "**/*.jsonl,**/*.json,**/*.parquet",
) -> Iterable[Dict[str, Any]]:
    """Yield records from local JSONL/JSON/Parquet files without changing data."""

    files = _resolve_local_files(dataset_path, dataset_format=dataset_format, file_glob=file_glob)
    if not files:
        raise ValueError(f"no eval data files found under {dataset_path}")

    parquet_files = []
    for path in files:
        file_format = _detect_dataset_file_format(path, dataset_format)
        if file_format == "jsonl":
            yield from _iter_jsonl_records(path)
        elif file_format == "json":
            yield from _iter_json_records(path)
        elif file_format == "parquet":
            parquet_files.append(path)
        else:
            raise ValueError(f"unsupported eval data file format for {path}")

    if parquet_files:
        yield from _iter_parquet_records(parquet_files)


def normalize_dataset_record(record: Dict[str, Any], *, dataset_id: str, index: int) -> EvalExample:
    """Convert supported math dataset rows into a prompt/answer pair."""

    prompt = extract_prompt(record)
    answer = extract_answer(record)
    if not prompt:
        raise ValueError(f"{dataset_id}:{index}: could not extract prompt")
    if not answer:
        raise ValueError(f"{dataset_id}:{index}: could not extract answer")

    stable_id_value = _first_present(record, ("uuid", "id", "problem_id", "raw_problem_id", "unique_id"))
    if stable_id_value is None:
        stable_id_value = _nested_get(record, ("extra_info", "index"))
    if stable_id_value is None or stable_id_value == "":
        stable_id_value = f"{dataset_id.replace('/', '__')}-{index:06d}"
    stable_id = str(stable_id_value)
    metadata = {
        key: record.get(key)
        for key in (
            "ability",
            "answer_type",
            "data_source",
            "source",
            "domain",
            "subject",
            "subfield",
            "type",
            "task_type",
            "difficulty",
            "level",
            "language",
            "is_multiple_answer",
            "question_type",
            "unit",
        )
        if key in record
    }
    return EvalExample(id=stable_id, prompt=prompt, answer=answer, metadata=metadata)


def extract_prompt(record: Dict[str, Any]) -> str:
    """Extract a user-visible problem statement from common math dataset schemas."""

    for key in ("raw_problem", "problem", "question", "input"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    prompt = record.get("prompt")
    text = _messages_to_text(prompt)
    if text:
        return text

    messages = record.get("messages")
    text = _messages_to_text(messages)
    if text:
        return text

    return ""


def extract_answer(record: Dict[str, Any]) -> str:
    """Extract a final answer field without using reasoning traces as prompts."""

    for key in ("answer", "final_answer", "target"):
        value = record.get(key)
        answer = _coerce_answer_text(value)
        if answer:
            return answer

    for path in (("verifier", "answer"), ("reward_model", "ground_truth")):
        answer = _coerce_answer_text(_nested_get(record, path))
        if answer:
            return answer

    solution = record.get("solution")
    answer = _coerce_answer_text(solution)
    if answer:
        return answer
    return ""


def format_math_prompt(problem: str, *, template: str = "boxed") -> str:
    """Use one stable boxed-answer instruction for all sampled datasets."""

    if template == "boxed":
        return (
            "Please solve the following math problem. Put your final answer in exactly "
            "one \\boxed{...}. Do not write anything after the boxed answer.\n\n"
            f"Problem:\n{problem}"
        )
    if template == "paper_math":
        return (
            "Solve the following math problem step by step. The last line of your "
            "response must contain exactly one final answer in \\boxed{...}, with "
            "nothing after the boxed answer.\n\n"
            f"{problem}"
        )
    raise ValueError(f"unsupported prompt template: {template}")


class DryRunMathGenerator:
    """Deterministic fake generator for tests and command smoke runs."""

    def generate_batch(self, prompts: List[str]) -> List[str]:
        return [r"\boxed{0}" for _ in prompts]

    def count_completion_tokens(self, completion: str) -> int:
        return len(completion.split())


class HFBatchedMathGenerator:
    """Small batched Hugging Face causal LM generator."""

    def __init__(self, config: Dict[str, Any]):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("torch and transformers are required for non-dry-run evals") from exc

        self.torch = torch
        self.config = config
        self.max_new_tokens = int(config.get("max_new_tokens") or 2048)
        self.apply_chat_template = not bool(config.get("no_chat_template", False))
        thinking_value = config.get("enable_thinking", "false")
        self.enable_thinking = None if thinking_value == "auto" else thinking_value == "true"
        model_name = config["model_name"]
        trust_remote_code = bool(config.get("trust_remote_code", False))

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        model_kwargs = {"trust_remote_code": trust_remote_code}
        dtype = _torch_dtype(config.get("torch_dtype"))
        if dtype is not None:
            model_kwargs["dtype"] = dtype
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.model.eval()

    def generate_batch(self, prompts: List[str]) -> List[str]:
        rendered = [self._render_prompt(prompt) for prompt in prompts]
        inputs = self.tokenizer(rendered, return_tensors="pt", padding=True)
        if hasattr(self.model, "device"):
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        temperature = float(self.config.get("temperature") or 0.0)
        do_sample = temperature > 0.0
        generation_kwargs = {
            "do_sample": do_sample,
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = float(self.config.get("top_p") or 1.0)

        with self.torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)

        prompt_length = inputs["input_ids"].shape[-1]
        completions = []
        for output in outputs:
            generated = output[prompt_length:]
            completions.append(self.tokenizer.decode(generated, skip_special_tokens=True))
        return completions

    def count_completion_tokens(self, completion: str) -> int:
        return len(self.tokenizer(completion, add_special_tokens=False)["input_ids"])

    def _render_prompt(self, prompt: str) -> str:
        if self.apply_chat_template:
            messages = [{"role": "user", "content": prompt}]
            kwargs = {}
            if self.enable_thinking is not None:
                kwargs["enable_thinking"] = self.enable_thinking
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **kwargs,
                )
            except (TypeError, ValueError):
                pass
        return prompt


def summarize_eval_rows(rows: Iterable[Dict[str, Any]], config: Optional[Dict[str, Any]] = None):
    rows = list(rows)
    total = len(rows)
    reward_values = [float(row["reward"]) for row in rows]
    full_credit_values = [float(row["reward"] == 1.0) for row in rows]
    parse_failures = [bool(row["parse_failed"]) for row in rows]
    parseable = [row for row in rows if not row["parse_failed"]]
    truncated = [row.get("truncated") for row in rows if row.get("truncated") is not None]
    reason_counts = Counter(str(row["reason"]) for row in rows)
    subject_counts = Counter(
        str(row.get("metadata", {}).get("subject") or row.get("metadata", {}).get("domain") or "unknown")
        for row in rows
    )

    return {
        "dataset_id": config.get("dataset_id") if config else None,
        "dataset_path": config.get("dataset_path") if config else None,
        "dataset_config": config.get("dataset_config") if config else None,
        "split": config.get("split") if config else None,
        "model_name": config.get("model_name") if config else None,
        "prompt_template": config.get("prompt_template") if config else None,
        "sample_size": total,
        "seed": config.get("seed") if config else None,
        "accuracy": _mean(full_credit_values),
        "full_credit_accuracy": _mean(full_credit_values),
        "reward_mean": _mean(reward_values),
        "format_success_rate": 1.0 - _mean(parse_failures) if total else None,
        "parse_failure_rate": _mean(parse_failures),
        "correctness_given_parse": _mean(float(row["reward"] == 1.0) for row in parseable),
        "avg_completion_chars": _mean(row["completion_chars"] for row in rows),
        "avg_completion_tokens": _mean(
            row["completion_tokens"] for row in rows if row.get("completion_tokens") is not None
        ),
        "truncation_rate": _mean(bool(value) for value in truncated) if truncated else None,
        "answer_mismatch_rate": reason_counts.get("answer_mismatch", 0) / total if total else None,
        "reason_counts": dict(sorted(reason_counts.items())),
        "subject_counts": dict(sorted(subject_counts.items())),
    }


def select_review_samples(rows: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    sample_count = int(config.get("save_sample_count") or 0)
    if sample_count <= 0:
        return []
    failures = [row for row in rows if row["reward"] == 0.0]
    correct = [row for row in rows if row["reward"] == 1.0]
    selected = failures[:sample_count]
    if len(selected) < sample_count:
        selected.extend(correct[: sample_count - len(selected)])
    return selected[:sample_count]


def _messages_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for message in value:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).lower()
            content = str(message.get("content", "")).strip()
            if role in {"user", ""} and content:
                parts.append(content)
        return "\n".join(parts).strip()
    return ""


def _coerce_answer_text(value: Any) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        parsed = _parse_stringified_scalar_list(stripped)
        return parsed if parsed is not None else stripped
    if isinstance(value, list):
        cleaned = [_coerce_answer_text(item) for item in value]
        cleaned = [item for item in cleaned if item]
        if not cleaned:
            return ""
        if len(cleaned) == 1:
            return cleaned[0]
        return ", ".join(cleaned)
    if value is not None and not isinstance(value, dict):
        return str(value).strip()
    return ""


def _parse_stringified_scalar_list(value: str) -> Optional[str]:
    if not (value.startswith("[") and value.endswith("]")):
        return None
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, list):
        return None
    cleaned = [_coerce_answer_text(item) for item in parsed]
    cleaned = [item for item in cleaned if item]
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) > 1:
        return ", ".join(cleaned)
    return ""


def _nested_get(record: Dict[str, Any], path: Iterable[str]) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _first_present(record: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key not in record:
            continue
        value = record[key]
        if value is not None and value != "":
            return value
    return None


def _resolve_local_files(dataset_path: Path, *, dataset_format: str, file_glob: str) -> List[Path]:
    if dataset_path.is_file():
        return [dataset_path]
    if not dataset_path.exists():
        raise FileNotFoundError(f"eval dataset path does not exist: {dataset_path}")
    patterns = [pattern.strip() for pattern in file_glob.split(",") if pattern.strip()]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(path for path in dataset_path.glob(pattern) if path.is_file())
    files = sorted(set(files))
    if dataset_format != "auto":
        files = [path for path in files if _detect_dataset_file_format(path, dataset_format) == dataset_format]
    return files


def _detect_dataset_file_format(path: Path, dataset_format: str) -> str:
    if dataset_format != "auto":
        return dataset_format
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".json":
        return "json"
    if suffix == ".parquet":
        return "parquet"
    return ""


def _iter_jsonl_records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield record


def _iter_json_records(path: Path) -> Iterable[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        payload = payload["data"]
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected a JSON list or object with a data list")
    for index, record in enumerate(payload, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{index}: expected JSON object")
        yield record


def _iter_parquet_records(paths: List[Path]) -> Iterable[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required to load local parquet eval files") from exc
    dataset = load_dataset("parquet", data_files=[str(path) for path in paths], split="train")
    for record in dataset:
        yield dict(record)


def _torch_dtype(name):
    if name in {None, "auto"}:
        return "auto" if name == "auto" else None
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for non-auto torch_dtype settings") from exc
    aliases = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in aliases:
        raise ValueError(f"unsupported torch_dtype: {name}")
    return aliases[name]


def _chunks(values: List[Any], size: int):
    size = max(1, size)
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _mean(values: Iterable[Any]):
    values = [float(value) for value in values]
    if not values:
        return None
    return sum(values) / len(values)


def _write_json(path: Path, payload: Dict[str, Any]):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_report(path: Path, summary: Dict[str, Any], config: Dict[str, Any]):
    lines = [
        "# Math Dataset Eval Report",
        "",
        f"- dataset: `{summary['dataset_id']}`",
        f"- dataset_path: `{summary.get('dataset_path')}`",
        f"- dataset_config: `{summary['dataset_config']}`",
        f"- split: `{summary['split']}`",
        f"- model: `{summary['model_name']}`",
        f"- prompt_template: `{summary.get('prompt_template')}`",
        f"- sample_size: `{summary['sample_size']}`",
        f"- seed: `{summary['seed']}`",
        f"- max_new_tokens: `{config.get('max_new_tokens')}`",
        f"- temperature: `{config.get('temperature')}`",
        f"- enable_thinking: `{config.get('enable_thinking')}`",
        "",
        "## Metrics",
        "",
    ]
    for key in (
        "accuracy",
        "format_success_rate",
        "parse_failure_rate",
        "correctness_given_parse",
        "answer_mismatch_rate",
        "truncation_rate",
        "avg_completion_tokens",
        "avg_completion_chars",
    ):
        lines.append(f"- {key}: {summary.get(key)}")
    lines.extend(["", "## Reason Counts", ""])
    for key, value in summary["reason_counts"].items():
        lines.append(f"- {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
