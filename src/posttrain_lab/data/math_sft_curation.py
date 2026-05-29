"""Curate parser-compatible boxed-answer math SFT data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


OPENR1_DATASET_ID = "open-r1/OpenR1-Math-220k"
DEEPMATH_DATASET_ID = "trl-lib/DeepMath-103K"
DEFAULT_OUTPUT_DIR = "data/staged/openr1_deepmath_sympy_boxed_v1"

BOOLEAN_RE = re.compile(r"^(yes|no|true|false)$", re.IGNORECASE)
CHOICE_RE = re.compile(r"^[A-E]$", re.IGNORECASE)
LOGICAL_WORD_RE = re.compile(r"\b(or|and)\b", re.IGNORECASE)
PROOF_PROMPT_RE = re.compile(
    r"\b(prove|show that|provide a justification|justify your answer|"
    r"is it possible|determine if|which is larger)\b",
    re.IGNORECASE,
)
BAD_PROMPT_RE = re.compile(
    r"(isomorphic|topolog|category|markov process|short exact sequence|"
    r"projective modules|uncountable set)",
    re.IGNORECASE,
)
LEAK_OR_MULTIPART_RE = re.compile(
    r"(\banswer\s*\.|questions?:|maximum and minimum values|"
    r"find the maximum and minimum|\(1\).+\(2\)|\ba\)\s|\bb\)\s|\bc\)\s|"
    r"!\[|https?://|translation preserves|translated|youth clue|figure below|"
    r"how many.+and how many)",
    re.IGNORECASE,
)
CONTROL_REPAIRS = {
    "\x0crac": r"\frac",
    "\x08oxed": r"\boxed",
}
ALLOWED_MACROS = {
    "alpha",
    "beta",
    "cos",
    "frac",
    "gamma",
    "lambda",
    "ln",
    "log",
    "mu",
    "pi",
    "sin",
    "sqrt",
    "tan",
    "theta",
}


@dataclass(frozen=True)
class CuratedSample:
    """One accepted parser-compatible math sample."""

    source: str
    source_id: str
    problem: str
    target: str
    domain: str


@dataclass(frozen=True)
class CurationDecision:
    """Decision for one raw problem/answer pair."""

    action: str
    reason: str
    clean_problem: str
    clean_target: str


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build parser-compatible boxed-answer SFT data.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-count", type=int, default=5000)
    parser.add_argument("--val-count", type=int, default=500)
    parser.add_argument("--test-count", type=int, default=500)
    parser.add_argument("--openr1-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--shuffle-buffer-size", type=int, default=10000)
    parser.add_argument("--max-source-records", type=int, default=250000)
    parser.add_argument("--eval-prompt-count", type=int, default=500)
    parser.add_argument("--no-parser", action="store_true", help="Skip latex2sympy2 parse gate.")
    args = parser.parse_args(argv)

    manifest = build_dataset(
        output_dir=args.output_dir,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
        openr1_ratio=args.openr1_ratio,
        seed=args.seed,
        shuffle_buffer_size=args.shuffle_buffer_size,
        max_source_records=args.max_source_records,
        eval_prompt_count=args.eval_prompt_count,
        require_parser=not args.no_parser,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


def build_dataset(
    output_dir,
    train_count=5000,
    val_count=500,
    test_count=500,
    openr1_ratio=0.7,
    seed=29,
    shuffle_buffer_size=10000,
    max_source_records=250000,
    eval_prompt_count=500,
    require_parser=True,
):
    """Build train/val/test SFT JSONL files and heldout eval prompts."""

    total_by_split = {"train": int(train_count), "val": int(val_count), "test": int(test_count)}
    source_quotas = _source_quotas(total_by_split, float(openr1_ratio))
    needed_by_source = {
        source: sum(split_counts.values()) for source, split_counts in source_quotas.items()
    }

    reject_counts = Counter()
    accepted_by_source = {
        OPENR1_DATASET_ID: collect_samples(
            source=OPENR1_DATASET_ID,
            needed=needed_by_source[OPENR1_DATASET_ID],
            seed=seed,
            shuffle_buffer_size=shuffle_buffer_size,
            max_source_records=max_source_records,
            require_parser=require_parser,
            reject_counts=reject_counts,
        ),
        DEEPMATH_DATASET_ID: collect_samples(
            source=DEEPMATH_DATASET_ID,
            needed=needed_by_source[DEEPMATH_DATASET_ID],
            seed=seed + 1,
            shuffle_buffer_size=shuffle_buffer_size,
            max_source_records=max_source_records,
            require_parser=require_parser,
            reject_counts=reject_counts,
        ),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    used_problem_keys = set()
    split_records = {"train": [], "val": [], "test": []}
    source_offsets = {OPENR1_DATASET_ID: 0, DEEPMATH_DATASET_ID: 0}
    source_counts_by_split = {}

    for split in ("train", "val", "test"):
        source_counts_by_split[split] = {}
        for source in (OPENR1_DATASET_ID, DEEPMATH_DATASET_ID):
            count = source_quotas[source][split]
            source_counts_by_split[split][source] = count
            samples = accepted_by_source[source][source_offsets[source] : source_offsets[source] + count]
            source_offsets[source] += count
            for sample in samples:
                key = _problem_key(sample.problem)
                if key in used_problem_keys:
                    raise ValueError(f"duplicate normalized problem across splits: {sample.source_id}")
                used_problem_keys.add(key)
                split_records[split].append(_to_sft_record(sample, split, len(split_records[split])))

    paths = {}
    for split, records in split_records.items():
        path = output_dir / f"{split}.jsonl"
        _write_jsonl(path, records)
        paths[f"{split}_path"] = str(path)

    eval_prompts = _to_eval_prompts(split_records["test"], limit=eval_prompt_count)
    eval_path = output_dir / "eval_prompts.jsonl"
    _write_jsonl(eval_path, eval_prompts)
    paths["eval_prompts_path"] = str(eval_path)

    manifest = {
        "dataset_version": output_dir.name,
        "sources": [OPENR1_DATASET_ID, DEEPMATH_DATASET_ID],
        "parser_required": bool(require_parser),
        "parser": _parser_metadata(require_parser),
        "split_counts": {split: len(records) for split, records in split_records.items()},
        "source_counts_by_split": source_counts_by_split,
        "eval_prompt_count": len(eval_prompts),
        "openr1_ratio": float(openr1_ratio),
        "seed": int(seed),
        "shuffle_buffer_size": int(shuffle_buffer_size),
        "max_source_records": int(max_source_records),
        "reject_counts": dict(sorted(reject_counts.items())),
        "paths": paths,
        "hashes": {name: _sha256_file(path) for name, path in paths.items()},
        "schema": "posttrain_lab SFT JSONL",
        "target_policy": "assistant message contains exactly one sanitized boxed parser-compatible final answer",
        "data_raw_modified": False,
    }
    manifest_path = output_dir / "manifest.json"
    _write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest["paths"]["manifest_path"] = str(manifest_path)
    manifest["hashes"]["manifest_path"] = _sha256_file(manifest_path)
    return manifest


def collect_samples(
    source,
    needed,
    seed,
    shuffle_buffer_size,
    max_source_records,
    require_parser,
    reject_counts,
):
    """Collect accepted samples from one Hugging Face dataset source."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required to build curated math SFT data") from exc

    dataset = load_dataset(source, split="train", streaming=True)
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(seed=int(seed), buffer_size=int(shuffle_buffer_size))

    accepted = []
    seen_problem_keys = set()
    for source_index, record in enumerate(dataset):
        if source_index >= max_source_records:
            break
        raw = _extract_source_record(source, record, source_index)
        if raw is None:
            reject_counts[f"{source}: unsupported_record"] += 1
            continue
        decision = curate_pair(raw["problem"], raw["answer"], require_parser=require_parser)
        if decision.action != "keep":
            reject_counts[f"{source}: {decision.reason}"] += 1
            continue
        key = _problem_key(decision.clean_problem)
        if key in seen_problem_keys:
            reject_counts[f"{source}: duplicate_problem"] += 1
            continue
        seen_problem_keys.add(key)
        accepted.append(
            CuratedSample(
                source=source,
                source_id=raw["source_id"],
                problem=decision.clean_problem,
                target=decision.clean_target,
                domain=raw["domain"],
            )
        )
        if len(accepted) == needed:
            break

    if len(accepted) != needed:
        raise ValueError(f"{source}: expected {needed} accepted samples, found {len(accepted)}")
    return accepted


def curate_pair(problem, target, require_parser=False):
    """Filter and sanitize one math problem/target pair."""

    clean_problem = _clean_problem(problem)
    if not clean_problem:
        return _reject("empty_problem")
    if len(clean_problem) > 900:
        return _reject("problem_too_long")
    if BAD_PROMPT_RE.search(clean_problem):
        return _reject("non_standard_or_abstract_prompt")
    if PROOF_PROMPT_RE.search(clean_problem):
        return _reject("proof_or_boolean_prompt")
    if LEAK_OR_MULTIPART_RE.search(clean_problem):
        return _reject("answer_leak_or_multipart_prompt")

    clean_target, reason = _sanitize_target(target)
    if clean_target is None:
        return _reject(reason)

    if require_parser:
        parse_reason = _parser_reject_reason(_unbox(clean_target))
        if parse_reason:
            return _reject(parse_reason)

    return CurationDecision(
        action="keep",
        reason="parser_compatible_boxed_final_answer",
        clean_problem=clean_problem,
        clean_target=clean_target,
    )


def _sanitize_target(target):
    target = _repair_control_latex(_clean_spaces(target))
    target = _strip_wrapping_box(target).strip().strip("$").strip()
    target = _strip_text_macros(target)
    target = _repair_control_latex(target)

    replacements = {
        r"\dfrac": r"\frac",
        r"\tfrac": r"\frac",
        r"\left": "",
        r"\right": "",
        r"\,": "",
        r"\;": "",
        r"\!": "",
        r"\cdoti": "i",
        r"\cdot i": "i",
        "~": "",
        "$": "",
        "°": "",
    }
    for old, new in replacements.items():
        target = target.replace(old, new)

    target = re.sub(r"\^\s*(?:\\circ|\\degree|\{\s*\})", "", target)
    target = re.sub(r"\\mathrm\{[^{}]*\}\s*/\s*\\mathrm\{[^{}]*\}", "", target)
    target = re.sub(r"\\(?:text|mathrm|rm|textbf)\s*\{[^{}]*\}", "", target)
    target = re.sub(r"\\(?:degree|circ)\b", "", target)
    target = target.replace("^{}", "")
    target = re.sub(r"\s+", " ", target).strip()
    if len(target) <= 120:
        target = target.replace(" ", "")

    if not target:
        return None, "empty_target"
    if BOOLEAN_RE.fullmatch(target):
        return None, "boolean_target"
    if CHOICE_RE.fullmatch(target):
        return None, "multiple_choice_letter_target"
    if LOGICAL_WORD_RE.search(target) or "or" in target.lower() or "and" in target.lower():
        return None, "logical_word_target"
    if "\\text" in target or "\\mathrm" in target or "\\begin" in target or "\\cases" in target:
        return None, "formatting_or_environment_target"
    if len(target) > 120:
        return None, "target_too_long"
    if any(ord(char) < 32 for char in target):
        return None, "control_character_target"

    macros = set(re.findall(r"\\([a-zA-Z]+)", target))
    unsupported = macros - ALLOWED_MACROS
    if unsupported:
        return None, "unsupported_macro_target"

    probe = re.sub(r"\\[a-zA-Z]+", "", target)
    if re.search(r"[A-Z]", probe):
        return None, "uppercase_text_target"
    if re.search(r"[a-zA-Z]{3,}", probe):
        return None, "word_like_target"

    return rf"\boxed{{{target}}}", ""


def _parser_reject_reason(target):
    try:
        from latex2sympy2 import latex2sympy
    except ImportError as exc:
        raise RuntimeError("latex2sympy2 is required when parser gate is enabled") from exc
    try:
        latex2sympy(target)
    except Exception:
        return "latex2sympy_parse_failure"
    return None


def _parser_metadata(require_parser):
    if not require_parser:
        return {"name": None, "version": None}
    try:
        import importlib.metadata as metadata
    except ImportError:
        import importlib_metadata as metadata
    try:
        version = metadata.version("latex2sympy2")
    except metadata.PackageNotFoundError:
        version = "unknown"
    return {"name": "latex2sympy2", "version": version}


def _extract_source_record(source, record, source_index):
    if source == OPENR1_DATASET_ID:
        problem = record.get("problem")
        answer = record.get("answer")
        if not isinstance(problem, str) or not isinstance(answer, str):
            return None
        return {
            "source_id": str(record.get("uuid") or f"openr1-{source_index:08d}"),
            "problem": problem,
            "answer": answer,
            "domain": str(record.get("problem_type") or record.get("question_type") or "math"),
        }
    if source == DEEPMATH_DATASET_ID:
        prompt = record.get("prompt")
        problem = ""
        if isinstance(prompt, list) and prompt:
            problem = prompt[0].get("content", "")
        elif isinstance(prompt, str):
            problem = prompt
        answer = record.get("solution")
        if not isinstance(problem, str) or not isinstance(answer, str):
            return None
        return {
            "source_id": f"deepmath-{source_index:08d}",
            "problem": problem,
            "answer": answer,
            "domain": "math",
        }
    return None


def _source_quotas(total_by_split, openr1_ratio):
    if not 0.0 <= openr1_ratio <= 1.0:
        raise ValueError("openr1_ratio must be between 0 and 1")
    quotas = {OPENR1_DATASET_ID: {}, DEEPMATH_DATASET_ID: {}}
    for split, total in total_by_split.items():
        openr1_count = round(total * openr1_ratio)
        quotas[OPENR1_DATASET_ID][split] = openr1_count
        quotas[DEEPMATH_DATASET_ID][split] = total - openr1_count
    return quotas


def _to_sft_record(sample, split, index):
    source_slug = "openr1" if sample.source == OPENR1_DATASET_ID else "deepmath"
    return {
        "id": f"sympy-boxed-v1-{source_slug}-{split}-{index:06d}",
        "split": split,
        "messages": [
            {"role": "user", "content": sample.problem},
            {"role": "assistant", "content": sample.target},
        ],
        "metadata": {
            "source": sample.source,
            "domain": sample.domain,
            "difficulty": "mixed",
            "license": "source-dataset-card",
        },
    }


def _to_eval_prompts(records, limit):
    rows = []
    for record in records[: int(limit)]:
        prompt = next(message["content"] for message in record["messages"] if message["role"] == "user")
        answer = next(message["content"] for message in record["messages"] if message["role"] == "assistant")
        rows.append({"id": record["id"], "prompt": prompt, "answer": answer})
    return rows


def _clean_problem(problem):
    return _clean_spaces(problem)


def _clean_spaces(text):
    text = _repair_control_latex(str(text or "")).replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"^#+\s*", "", text)


def _strip_wrapping_box(text):
    match = re.fullmatch(r"\\boxed\s*\{(.*)\}", text)
    return match.group(1).strip() if match else text


def _strip_text_macros(text):
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\\(?:text|mathrm|rm|textbf)\s*\{([^{}]*)\}", r"\1", text)
    return text


def _repair_control_latex(text):
    for bad, good in CONTROL_REPAIRS.items():
        text = text.replace(bad, good)
    return text


def _unbox(clean_target):
    return clean_target[len(r"\boxed{") : -1]


def _reject(reason):
    return CurationDecision(action="reject", reason=reason, clean_problem="", clean_target="")


def _problem_key(problem):
    return re.sub(r"\s+", " ", problem.strip().lower())


def _write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_text(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
