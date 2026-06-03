"""Thin verl entrypoints for boxed math rewards."""

from __future__ import annotations

from typing import Any

from posttrain_lab.rewards.math_reward import (
    compute_score as _compute_score,
    compute_score_verl_style as _compute_score_verl_style,
)


def compute_score(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Forward verl reward calls to ``math_boxed_v001`` without changing semantics."""

    result = dict(_compute_score(*args, **kwargs))
    return _sanitize_extra_info(result)


def compute_score_verl_style(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Forward verl reward calls to ``math_boxed_verl_v001``."""

    result = dict(_compute_score_verl_style(*args, **kwargs))
    return _sanitize_extra_info(result)


def compute_score_common(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias for configs that prefer a shorter common-verl-style name."""

    return compute_score_verl_style(*args, **kwargs)


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
