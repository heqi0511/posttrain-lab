"""CLI and library validators for SFT and RLVR JSONL files."""

import argparse
import json
import sys
from pathlib import Path

from posttrain_lab.data.schemas import (
    MESSAGE_REQUIRED_FIELDS,
    METADATA_REQUIRED_FIELDS,
    RLVR_REQUIRED_FIELDS,
    SFT_MESSAGE_ROLES,
    SFT_REQUIRED_FIELDS,
    VALID_SPLITS,
    VERIFIER_REQUIRED_FIELDS,
    ValidationError,
)


def validate_jsonl(dataset_type, path):
    """Validate an SFT or RLVR JSONL file and return validation errors."""

    path = Path(path)
    errors = []
    seen_ids = {}

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                errors.append(_error(path, line_number, "empty line is not allowed"))
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(_error(path, line_number, f"invalid JSON: {exc.msg}"))
                continue

            if not isinstance(record, dict):
                errors.append(_error(path, line_number, "record must be an object"))
                continue

            if dataset_type == "sft":
                errors.extend(_validate_sft_record(record, path, line_number))
            elif dataset_type == "rlvr":
                errors.extend(_validate_rlvr_record(record, path, line_number))
            else:
                errors.append(_error(path, line_number, f"unknown dataset type: {dataset_type}"))
                continue

            record_id = record.get("id")
            if isinstance(record_id, str) and record_id:
                if record_id in seen_ids:
                    first_line = seen_ids[record_id]
                    errors.append(
                        _error(
                            path,
                            line_number,
                            f"duplicate id: {record_id} first seen on line {first_line}",
                        )
                    )
                else:
                    seen_ids[record_id] = line_number

    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate SFT or RLVR JSONL data.")
    parser.add_argument("--type", choices=("sft", "rlvr"), required=True)
    parser.add_argument("--path", required=True)
    args = parser.parse_args(argv)

    errors = validate_jsonl(args.type, args.path)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"{args.path}: valid")
    return 0


def _validate_sft_record(record, path, line_number):
    errors = []
    errors.extend(_validate_object_fields(record, SFT_REQUIRED_FIELDS, path, line_number))
    errors.extend(_validate_common_fields(record, path, line_number))

    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        errors.append(_error(path, line_number, "messages must be a non-empty list"))
        return errors

    has_assistant = False
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(_error(path, line_number, f"messages[{index}] must be an object"))
            continue
        errors.extend(
            _validate_object_fields(message, MESSAGE_REQUIRED_FIELDS, path, line_number, f"messages[{index}]")
        )
        role = message.get("role")
        if role not in SFT_MESSAGE_ROLES:
            errors.append(_error(path, line_number, f"invalid messages[{index}].role: {role}"))
        if role == "assistant":
            has_assistant = True
        _validate_non_empty_string(message.get("content"), f"messages[{index}].content", path, line_number, errors)

    if not has_assistant:
        errors.append(_error(path, line_number, "messages must include at least one assistant message"))

    return errors


def _validate_rlvr_record(record, path, line_number):
    errors = []
    errors.extend(_validate_object_fields(record, RLVR_REQUIRED_FIELDS, path, line_number))
    errors.extend(_validate_common_fields(record, path, line_number))

    prompt = record.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        errors.append(_error(path, line_number, "prompt must be a non-empty list"))
    else:
        for index, message in enumerate(prompt):
            if not isinstance(message, dict):
                errors.append(_error(path, line_number, f"prompt[{index}] must be an object"))
                continue
            errors.extend(
                _validate_object_fields(message, MESSAGE_REQUIRED_FIELDS, path, line_number, f"prompt[{index}]")
            )
            role = message.get("role")
            if role != "user":
                errors.append(_error(path, line_number, f"RLVR prompt[{index}].role must be user, got {role}"))
            _validate_non_empty_string(message.get("content"), f"prompt[{index}].content", path, line_number, errors)

    verifier = record.get("verifier")
    if not isinstance(verifier, dict):
        errors.append(_error(path, line_number, "verifier must be an object"))
    else:
        errors.extend(_validate_object_fields(verifier, VERIFIER_REQUIRED_FIELDS, path, line_number, "verifier"))
        _validate_non_empty_string(verifier.get("type"), "verifier.type", path, line_number, errors)
        _validate_non_empty_string(verifier.get("answer"), "verifier.answer", path, line_number, errors)

    return errors


def _validate_common_fields(record, path, line_number):
    errors = []
    _validate_non_empty_string(record.get("id"), "id", path, line_number, errors)

    split = record.get("split")
    if split not in VALID_SPLITS:
        errors.append(_error(path, line_number, f"invalid split: {split}"))

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        errors.append(_error(path, line_number, "metadata must be an object"))
    else:
        errors.extend(_validate_object_fields(metadata, METADATA_REQUIRED_FIELDS, path, line_number, "metadata"))
        for field in sorted(METADATA_REQUIRED_FIELDS):
            _validate_non_empty_string(metadata.get(field), f"metadata.{field}", path, line_number, errors)

    return errors


def _validate_object_fields(obj, required_fields, path, line_number, prefix=None):
    errors = []
    actual_fields = set(obj)
    for field in sorted(required_fields - actual_fields):
        name = f"{prefix}.{field}" if prefix else field
        errors.append(_error(path, line_number, f"missing required field: {name}"))
    for field in sorted(actual_fields - required_fields):
        name = f"{prefix}.{field}" if prefix else field
        errors.append(_error(path, line_number, f"unexpected field: {name}"))
    return errors


def _validate_non_empty_string(value, field_name, path, line_number, errors):
    if not isinstance(value, str) or not value.strip():
        errors.append(_error(path, line_number, f"{field_name} must be non-empty"))


def _error(path, line_number, message):
    return ValidationError(Path(path), line_number, message)


if __name__ == "__main__":
    raise SystemExit(main())
