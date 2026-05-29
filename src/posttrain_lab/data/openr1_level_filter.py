"""Stage OpenR1-Math level/domain-filtered RLVR prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from posttrain_lab.data.math_sft_curation import OPENR1_DATASET_ID, curate_pair


DEFAULT_OUTPUT_DIR = "data/rlvr_prompts/openr1_math_l2_l3_alg_nt_v1"
DEFAULT_LEVELS = (2, 3)
DEFAULT_DOMAINS = ("Algebra", "Number Theory")
BOXED_SUFFIX = "\n\nReturn only the final answer in exactly one \\boxed{...}."


@dataclass(frozen=True)
class FilteredPrompt:
    """One filtered OpenR1-Math prompt suitable for RLVR."""

    source_id: str
    problem: str
    answer: str
    domain: str
    level: int


def main(argv=None):
    parser = argparse.ArgumentParser(description="Filter OpenR1-Math by level/domain into RLVR JSONL.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-id", default=OPENR1_DATASET_ID)
    parser.add_argument("--dataset-config", default="default")
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--levels", default="2,3")
    parser.add_argument("--domains", default="Algebra,Number Theory")
    parser.add_argument("--train-count", type=int, default=5000)
    parser.add_argument("--val-count", type=int, default=500)
    parser.add_argument("--test-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--shuffle-buffer-size", type=int, default=10000)
    parser.add_argument("--max-source-records", type=int, default=250000)
    parser.add_argument("--no-parser", action="store_true", help="Skip the existing latex2sympy2 curation gate.")
    args = parser.parse_args(argv)

    manifest = build_openr1_level_filtered_dataset(
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        dataset_config=args.dataset_config,
        source_split=args.source_split,
        levels=_parse_levels(args.levels),
        domains=_parse_domains(args.domains),
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
        seed=args.seed,
        shuffle_buffer_size=args.shuffle_buffer_size,
        max_source_records=args.max_source_records,
        require_parser=not args.no_parser,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


def build_openr1_level_filtered_dataset(
    output_dir,
    dataset_id=OPENR1_DATASET_ID,
    dataset_config="default",
    source_split="train",
    levels=DEFAULT_LEVELS,
    domains=DEFAULT_DOMAINS,
    train_count=5000,
    val_count=500,
    test_count=500,
    seed=31,
    shuffle_buffer_size=10000,
    max_source_records=250000,
    require_parser=True,
):
    """Build level/domain-filtered RLVR train/val/test JSONL files."""

    total_needed = int(train_count) + int(val_count) + int(test_count)
    reject_counts = Counter()
    source_records = _load_openr1_records(
        dataset_id=dataset_id,
        dataset_config=dataset_config,
        source_split=source_split,
        seed=seed,
        shuffle_buffer_size=shuffle_buffer_size,
    )
    prompts = collect_filtered_prompts(
        source_records,
        needed=total_needed,
        levels=levels,
        domains=domains,
        max_source_records=max_source_records,
        require_parser=require_parser,
        reject_counts=reject_counts,
    )
    if len(prompts) != total_needed:
        missing_level_count = reject_counts.get("missing_level", 0)
        hint = ""
        if missing_level_count:
            hint = (
                " The source records appear to be missing a level field; "
                "OpenR1-Math-220k currently exposes problem_type but not always level."
            )
        raise ValueError(f"expected {total_needed} filtered prompts, found {len(prompts)}.{hint}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_prompts = {
        "train": prompts[: int(train_count)],
        "val": prompts[int(train_count) : int(train_count) + int(val_count)],
        "test": prompts[int(train_count) + int(val_count) :],
    }
    paths = {}
    for split, rows in split_prompts.items():
        path = output_dir / f"{split}.jsonl"
        _write_jsonl(path, [_to_rlvr_record(row, split, index) for index, row in enumerate(rows)])
        paths[f"{split}_path"] = str(path)

    manifest = {
        "dataset_version": output_dir.name,
        "dataset_id": dataset_id,
        "dataset_config": dataset_config,
        "source_split": source_split,
        "levels": [int(level) for level in levels],
        "domains": list(domains),
        "split_counts": {split: len(rows) for split, rows in split_prompts.items()},
        "seed": int(seed),
        "shuffle_buffer_size": int(shuffle_buffer_size),
        "max_source_records": int(max_source_records),
        "parser_required": bool(require_parser),
        "reject_counts": dict(sorted(reject_counts.items())),
        "paths": paths,
        "hashes": {name: _sha256_file(path) for name, path in paths.items()},
        "schema": "posttrain_lab RLVR JSONL",
        "data_raw_modified": False,
    }
    manifest_path = output_dir / "manifest.json"
    _write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest["paths"]["manifest_path"] = str(manifest_path)
    manifest["hashes"]["manifest_path"] = _sha256_file(manifest_path)
    return manifest


def collect_filtered_prompts(
    records,
    needed,
    levels=DEFAULT_LEVELS,
    domains=DEFAULT_DOMAINS,
    max_source_records=250000,
    require_parser=True,
    reject_counts=None,
):
    """Collect filtered, parser-compatible prompts from OpenR1-like records."""

    reject_counts = reject_counts if reject_counts is not None else Counter()
    allowed_levels = {int(level) for level in levels}
    allowed_domains = {_normalize_label(domain) for domain in domains}
    accepted = []
    seen_problem_keys = set()

    for source_index, record in enumerate(records):
        if source_index >= int(max_source_records):
            break

        extracted = extract_openr1_level_record(record, source_index)
        if extracted is None:
            reject_counts["unsupported_record"] += 1
            continue
        if extracted["level"] is None:
            reject_counts["missing_level"] += 1
            continue
        if int(extracted["level"]) not in allowed_levels:
            reject_counts["level_not_selected"] += 1
            continue
        if _normalize_label(extracted["domain"]) not in allowed_domains:
            reject_counts["domain_not_selected"] += 1
            continue

        decision = curate_pair(extracted["problem"], extracted["answer"], require_parser=require_parser)
        if decision.action != "keep":
            reject_counts[decision.reason] += 1
            continue

        key = _problem_key(decision.clean_problem)
        if key in seen_problem_keys:
            reject_counts["duplicate_problem"] += 1
            continue
        seen_problem_keys.add(key)
        accepted.append(
            FilteredPrompt(
                source_id=extracted["source_id"],
                problem=decision.clean_problem,
                answer=_unbox(decision.clean_target),
                domain=extracted["domain"],
                level=int(extracted["level"]),
            )
        )
        if len(accepted) == int(needed):
            break
    return accepted


def extract_openr1_level_record(record, source_index=0):
    """Extract problem, answer, level, and domain from an OpenR1-like record."""

    problem = record.get("problem")
    answer = record.get("answer")
    if not isinstance(problem, str) or not isinstance(answer, str):
        return None
    domain = record.get("problem_type") or record.get("domain") or record.get("subject") or ""
    return {
        "source_id": str(record.get("uuid") or f"openr1-{source_index:08d}"),
        "problem": problem,
        "answer": answer,
        "domain": str(domain),
        "level": _extract_level(record),
    }


def _load_openr1_records(dataset_id, dataset_config, source_split, seed, shuffle_buffer_size):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required to build OpenR1 filtered RLVR data") from exc

    if dataset_config:
        dataset = load_dataset(dataset_id, dataset_config, split=source_split, streaming=True)
    else:
        dataset = load_dataset(dataset_id, split=source_split, streaming=True)
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(seed=int(seed), buffer_size=int(shuffle_buffer_size))
    return dataset


def _extract_level(record):
    candidates = [
        record.get("level"),
        record.get("difficulty_level"),
        record.get("problem_level"),
        record.get("math_level"),
        record.get("metadata", {}).get("level") if isinstance(record.get("metadata"), dict) else None,
    ]
    for value in candidates:
        parsed = _parse_level(value)
        if parsed is not None:
            return parsed
    return None


def _parse_level(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip()
    match = re.fullmatch(r"(?:level\s*)?([0-9]+)", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _to_rlvr_record(prompt, split, index):
    domain = _canonical_domain(prompt.domain)
    return {
        "id": f"openr1-l23-{_slug(domain)}-{split}-{index:06d}",
        "split": split,
        "prompt": [
            {
                "role": "user",
                "content": prompt.problem + BOXED_SUFFIX,
            }
        ],
        "verifier": {
            "type": "math_boxed_v001",
            "answer": prompt.answer,
        },
        "metadata": {
            "source": OPENR1_DATASET_ID,
            "domain": domain,
            "difficulty": f"level_{prompt.level}",
            "license": "source-dataset-card",
        },
    }


def _parse_levels(value):
    return tuple(int(item.strip()) for item in str(value).split(",") if item.strip())


def _parse_domains(value):
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _normalize_label(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _canonical_domain(value):
    normalized = _normalize_label(value)
    for domain in DEFAULT_DOMAINS:
        if _normalize_label(domain) == normalized:
            return domain
    return str(value).strip()


def _slug(value):
    return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")


def _problem_key(problem):
    return re.sub(r"\s+", " ", problem.strip().lower())


def _unbox(clean_target):
    return clean_target[len(r"\boxed{") : -1]


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
