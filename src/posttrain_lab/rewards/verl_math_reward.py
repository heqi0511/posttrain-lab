"""Thin verl entrypoint for the strict boxed math reward."""

from __future__ import annotations

from typing import Any

from posttrain_lab.rewards.math_reward import compute_score as _compute_score


def compute_score(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Forward verl reward calls to ``math_boxed_v001`` without changing semantics."""

    result = dict(_compute_score(*args, **kwargs))
    return _sanitize_extra_info(result)


def _sanitize_extra_info(result: dict[str, Any]) -> dict[str, Any]:
    """Keep verl validation metric aggregation away from ``None`` metadata.

    verl aggregates non-string reward extra fields during validation. The strict
    boxed reward uses ``None`` for missing parser details, which is useful for
    local debugging but crashes that aggregation path. This adapter keeps the
    scalar score unchanged and only normalizes metadata for verl.
    """

    for key in ("reason", "reward_version", "extracted_answer", "normalized_prediction", "normalized_answer", "data_source"):
        if result.get(key) is None:
            result[key] = ""
        else:
            result[key] = str(result[key])
    if result.get("ground_truth_index") is None:
        result["ground_truth_index"] = -1
    return result
