"""Filter existing SFT/RLVR JSONL directories with a target policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from posttrain_lab.data.math_sft_curation import (
    DEFAULT_TARGET_POLICY,
    SINGLE_EXPRESSION_NO_ASSIGNMENT_POLICY,
    target_policy_reject_reason,
)
from posttrain_lab.data.validate import validate_jsonl
from posttrain_lab.rewards.math_reward import extract_boxed_answers


SPLITS = ("train", "val", "test")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Filter staged RLVR/SFT data by target policy.")
    parser.add_argument("--input-rlvr-dir", required=True)
    parser.add_argument("--output-rlvr-dir", required=True)
    parser.add_argument("--input-sft-dir", default="")
    parser.add_argument("--output-sft-dir", default="")
    parser.add_argument("--target-policy", default=SINGLE_EXPRESSION_NO_ASSIGNMENT_POLICY)
    args = parser.parse_args(argv)

    manifest = filter_dataset_dirs(
        input_rlvr_dir=args.input_rlvr_dir,
        output_rlvr_dir=args.output_rlvr_dir,
        input_sft_dir=args.input_sft_dir,
        output_sft_dir=args.output_sft_dir,
        target_policy=args.target_policy,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


def filter_dataset_dirs(
    input_rlvr_dir,
    output_rlvr_dir,
    input_sft_dir="",
    output_sft_dir="",
    target_policy=DEFAULT_TARGET_POLICY,
):
    """Filter existing split JSONL files while preserving split membership."""

    input_rlvr_dir = Path(input_rlvr_dir)
    output_rlvr_dir = Path(output_rlvr_dir)
    output_rlvr_dir.mkdir(parents=True, exist_ok=True)

    rlvr_result = _filter_split_dir(
        input_dir=input_rlvr_dir,
        output_dir=output_rlvr_dir,
        dataset_type="rlvr",
        target_policy=target_policy,
    )
    manifest = _manifest(
        input_dir=input_rlvr_dir,
        output_dir=output_rlvr_dir,
        dataset_type="rlvr",
        target_policy=target_policy,
        result=rlvr_result,
    )

    if input_sft_dir and output_sft_dir:
        input_sft_dir = Path(input_sft_dir)
        output_sft_dir = Path(output_sft_dir)
        output_sft_dir.mkdir(parents=True, exist_ok=True)
        sft_result = _filter_split_dir(
            input_dir=input_sft_dir,
            output_dir=output_sft_dir,
            dataset_type="sft",
            target_policy=target_policy,
        )
        sft_manifest = _manifest(
            input_dir=input_sft_dir,
            output_dir=output_sft_dir,
            dataset_type="sft",
            target_policy=target_policy,
            result=sft_result,
        )
        _write_manifest(output_sft_dir / "manifest.json", sft_manifest)
        manifest["paired_sft_output_dir"] = str(output_sft_dir)
        manifest["paired_sft_manifest_path"] = str(output_sft_dir / "manifest.json")

    _write_manifest(output_rlvr_dir / "manifest.json", manifest)
    return manifest


def _filter_split_dir(input_dir, output_dir, dataset_type, target_policy):
    split_counts = {}
    input_counts = {}
    reject_counts = Counter()
    paths = {}

    for split in SPLITS:
        input_path = input_dir / f"{split}.jsonl"
        output_path = output_dir / f"{split}.jsonl"
        if not input_path.exists():
            continue
        errors = validate_jsonl(dataset_type, input_path)
        if errors:
            raise ValueError("\n".join(str(error) for error in errors))

        kept = []
        total = 0
        for record in _read_jsonl(input_path):
            total += 1
            reason = _record_reject_reason(record, dataset_type, target_policy)
            if reason:
                reject_counts[reason] += 1
                continue
            kept.append(record)

        _write_jsonl(output_path, kept)
        errors = validate_jsonl(dataset_type, output_path)
        if errors:
            raise ValueError("\n".join(str(error) for error in errors))
        input_counts[split] = total
        split_counts[split] = len(kept)
        paths[f"{split}_path"] = str(output_path)

    return {
        "input_counts": input_counts,
        "split_counts": split_counts,
        "reject_counts": dict(sorted(reject_counts.items())),
        "paths": paths,
    }


def _record_reject_reason(record, dataset_type, target_policy):
    if dataset_type == "rlvr":
        target = record["verifier"]["answer"]
    elif dataset_type == "sft":
        assistants = [message for message in record["messages"] if message["role"] == "assistant"]
        if len(assistants) != 1:
            return "assistant_message_count"
        boxed = extract_boxed_answers(assistants[0]["content"])
        if len(boxed) != 1:
            return "assistant_boxed_count"
        target = boxed[0]
    else:
        raise ValueError(f"unsupported dataset_type: {dataset_type}")
    return target_policy_reject_reason(target, target_policy)


def _manifest(input_dir, output_dir, dataset_type, target_policy, result):
    paths = dict(result["paths"])
    manifest_path = output_dir / "manifest.json"
    paths["manifest_path"] = str(manifest_path)
    hashes = {
        name: _sha256_file(path)
        for name, path in paths.items()
        if name != "manifest_path" and Path(path).exists()
    }
    return {
        "dataset_version": output_dir.name,
        "schema": f"posttrain_lab {dataset_type.upper()} JSONL",
        "source_input_dir": str(input_dir),
        "target_policy_name": str(target_policy),
        "target_policy": _target_policy_description(target_policy),
        "input_counts": result["input_counts"],
        "split_counts": result["split_counts"],
        "reject_counts": result["reject_counts"],
        "paths": paths,
        "hashes": hashes,
        "split_membership_preserved": True,
        "data_raw_modified": False,
    }


def _target_policy_description(target_policy):
    if target_policy == SINGLE_EXPRESSION_NO_ASSIGNMENT_POLICY:
        return "Reject targets containing assignment/equation '=' or plus-minus macros before SFT/RLVR use."
    return "Default parser-compatible boxed target policy."


def _write_manifest(path, manifest):
    manifest = dict(manifest)
    manifest["hashes"] = dict(manifest.get("hashes", {}))
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["hashes"]["manifest_path"] = _sha256_file(path)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
