"""Thin verl entrypoint for the strict boxed math reward."""

from __future__ import annotations

from typing import Any

from posttrain_lab.rewards.math_reward import compute_score as _compute_score


def compute_score(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Forward verl reward calls to ``math_boxed_v001`` without changing semantics."""

    return _compute_score(*args, **kwargs)
