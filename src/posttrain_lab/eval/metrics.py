"""Small deterministic metrics for local eval runs."""

import re


def exact_match(generation, answer):
    """Return whether generation exactly matches answer after outer whitespace trim."""

    return generation.strip() == answer.strip()


def format_success(pattern, generation):
    """Return whether generation satisfies the configured output format regex."""

    if not pattern:
        return None
    return re.search(pattern, generation.strip()) is not None


def mean_boolean(values):
    """Average boolean values, returning None for an empty list."""

    if not values:
        return None
    return sum(1 for value in values if value) / len(values)
