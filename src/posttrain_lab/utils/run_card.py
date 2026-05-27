"""Reusable run-card writer for experiment metadata."""

from __future__ import annotations

import json
from pathlib import Path


REQUIRED_RUN_CARD_FIELDS = {
    "base_model",
    "output_path",
    "git_commit",
    "data_path",
    "data_hash",
    "config_path",
    "config_hash",
    "eval_version",
    "final_metrics",
}


def write_run_card(path, metadata):
    """Write a Markdown run card after validating required metadata."""

    missing = sorted(REQUIRED_RUN_CARD_FIELDS - set(metadata))
    if missing:
        raise ValueError(f"missing required run-card metadata: {', '.join(missing)}")

    lines = ["# Run Card", ""]
    for field in sorted(REQUIRED_RUN_CARD_FIELDS):
        lines.append(f"- {field}: `{_format_value(metadata[field])}`")

    optional_fields = sorted(set(metadata) - REQUIRED_RUN_CARD_FIELDS)
    if optional_fields:
        lines.extend(["", "## Optional Metadata", ""])
        for field in optional_fields:
            lines.append(f"- {field}: `{_format_value(metadata[field])}`")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)
