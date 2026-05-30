"""Small deterministic metrics for local eval runs."""

import re


def exact_match(generation, answer):
    """Return whether generation exactly matches answer after outer whitespace trim."""

    return generation.strip() == answer.strip()


def format_success(pattern, generation):
    """Return whether generation satisfies the configured output format regex."""

    if not pattern:
        return None
    return re.search(_normalize_format_pattern(pattern), generation.strip()) is not None


def _normalize_format_pattern(pattern):
    """Normalize common YAML regex intent for literal LaTeX boxed answers."""

    return _escape_single_backslash_boxed(pattern)


def _escape_single_backslash_boxed(pattern):
    result = []
    index = 0
    token = r"\boxed"
    while index < len(pattern):
        if pattern.startswith(token, index) and (index == 0 or pattern[index - 1] != "\\"):
            result.append(r"\\boxed")
            index += len(token)
            continue
        result.append(pattern[index])
        index += 1
    return "".join(result)


def regex_format_success(text, pattern):
    """Compatibility wrapper for regex-based format checks."""

    return format_success(pattern, text)


def mean_boolean(values):
    """Average boolean values, returning None for an empty list."""

    values = list(values)
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def compute_eval_metrics(rows):
    """Compute stable aggregate metrics from eval result rows."""

    rows = list(rows)
    if not rows:
        return {
            "num_examples": 0,
            "accuracy": None,
            "format_success_rate": None,
            "parse_failure_rate": None,
            "avg_output_length": None,
        }

    return {
        "num_examples": len(rows),
        "accuracy": _mean_numeric(row.get("score") for row in rows),
        "format_success_rate": mean_boolean(row.get("format_success") for row in rows),
        "parse_failure_rate": mean_boolean(row.get("parse_failed") for row in rows),
        "avg_output_length": _mean_numeric(row.get("output_length") for row in rows),
    }


def _mean_numeric(values):
    filtered = [value for value in values if isinstance(value, (int, float))]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)
