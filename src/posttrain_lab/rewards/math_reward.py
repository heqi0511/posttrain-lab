"""Deterministic boxed-answer math reward."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Tuple


REWARD_VERSION = "math_boxed_v001"
_BOXED = r"\boxed"
_MAX_SYMBOLIC_EXPR_CHARS = 120
_MAX_SYMBOLIC_AST_NODES = 64
_MAX_FRACTION_COMPONENT = 10**12


@dataclass(frozen=True)
class MathRewardConfig:
    """Configuration for math_boxed_v001."""

    allow_symbolic_equivalence: bool = False
    max_output_chars: int = 20_000
    max_answer_chars: int = 1_000


@dataclass(frozen=True)
class MathRewardResult:
    """Structured reward result for debugging and run cards."""

    score: float
    reason: str
    version: str = REWARD_VERSION
    extracted_answer: Optional[str] = None
    normalized_prediction: Optional[str] = None
    normalized_answer: Optional[str] = None


@dataclass(frozen=True)
class RewardResult:
    """Compatibility reward result for starter test-suite APIs."""

    reward: float
    parsed_answer: Optional[str]
    failure_reason: Optional[str]


def math_boxed_v001(
    completion: str,
    answer: str,
    *,
    config: Optional[MathRewardConfig] = None,
    allow_symbolic_equivalence: Optional[bool] = None,
) -> float:
    """Return the scalar math_boxed_v001 reward."""

    if allow_symbolic_equivalence is not None:
        base = config or MathRewardConfig()
        config = MathRewardConfig(
            allow_symbolic_equivalence=allow_symbolic_equivalence,
            max_output_chars=base.max_output_chars,
            max_answer_chars=base.max_answer_chars,
        )
    return score_math_boxed_v001(completion, answer, config=config).score


def score_math_boxed(completion: str, expected_answer: str, symbolic: bool = False) -> RewardResult:
    """Compatibility wrapper that keeps the strict v001 reward unchanged.

    The project training path uses ``math_boxed_v001``. This wrapper exists for
    older tests and tools that score any well-formed boxed answer and allow
    repeated identical boxed answers.
    """

    if _has_unclosed_think_block(completion):
        return RewardResult(0.0, None, "unclosed_think_block")

    scored_completion = _strip_think_blocks(completion)
    boxed_answers, malformed = _extract_boxed_answers_with_status(scored_completion)
    if malformed:
        return RewardResult(0.0, None, "malformed_boxed_answer")
    if not boxed_answers:
        return RewardResult(0.0, None, "no_boxed_answer")

    normalized_boxes = [normalize_math_answer(value) for value in boxed_answers]
    if len(set(normalized_boxes)) > 1:
        return RewardResult(0.0, normalized_boxes[-1], "conflicting_boxed_answers")

    parsed_answer = normalized_boxes[-1]
    normalized_expected = normalize_math_answer(expected_answer)
    if parsed_answer == normalized_expected:
        return RewardResult(1.0, parsed_answer, None)
    if symbolic and _symbolically_equivalent(parsed_answer, normalized_expected):
        return RewardResult(1.0, parsed_answer, None)
    return RewardResult(0.0, parsed_answer, "answer_mismatch")


def score_math_boxed_v001(
    completion: str,
    answer: str,
    *,
    config: Optional[MathRewardConfig] = None,
) -> MathRewardResult:
    """Score a model completion against a reference answer.

    The reward is binary: 1.0 for a valid boxed answer matching the reference,
    0.0 otherwise. It is offline, deterministic, and scans only bounded text.
    """

    config = config or MathRewardConfig()
    if len(completion) > config.max_output_chars:
        return MathRewardResult(score=0.0, reason="output_too_long")
    if len(answer) > config.max_answer_chars:
        return MathRewardResult(score=0.0, reason="answer_too_long")
    if _has_unclosed_think_block(completion):
        return MathRewardResult(score=0.0, reason="unclosed_think_block")

    scored_completion = _strip_think_blocks(completion)
    boxed_answers, malformed = _extract_boxed_answers_with_status(scored_completion)
    if malformed:
        return MathRewardResult(score=0.0, reason="malformed_boxed_answer")
    if not boxed_answers:
        return MathRewardResult(score=0.0, reason="no_boxed_answer")
    if len(boxed_answers) > 1:
        normalized_boxes = [normalize_math_answer(value) for value in boxed_answers]
        if len(set(normalized_boxes)) > 1:
            return MathRewardResult(score=0.0, reason="conflicting_boxed_answers")
        return MathRewardResult(score=0.0, reason="multiple_boxed_answers")
    if not _boxed_is_final_only(scored_completion):
        return MathRewardResult(score=0.0, reason="boxed_not_final_only")

    normalized_boxes = [normalize_math_answer(value) for value in boxed_answers]
    if any(not value for value in normalized_boxes):
        return MathRewardResult(score=0.0, reason="empty_boxed_answer")

    normalized_prediction = normalized_boxes[0]
    normalized_answer = normalize_math_answer(answer)
    if normalized_prediction == normalized_answer:
        return MathRewardResult(
            score=1.0,
            reason="exact_match",
            extracted_answer=boxed_answers[-1],
            normalized_prediction=normalized_prediction,
            normalized_answer=normalized_answer,
        )

    if config.allow_symbolic_equivalence and _symbolically_equivalent(
        normalized_prediction, normalized_answer
    ):
        return MathRewardResult(
            score=1.0,
            reason="symbolic_equivalence",
            extracted_answer=boxed_answers[-1],
            normalized_prediction=normalized_prediction,
            normalized_answer=normalized_answer,
        )

    return MathRewardResult(
        score=0.0,
        reason="answer_mismatch",
        extracted_answer=boxed_answers[-1],
        normalized_prediction=normalized_prediction,
        normalized_answer=normalized_answer,
    )


def extract_boxed_answers(text: str) -> List[str]:
    """Extract well-formed boxed answer payloads."""

    answers, _ = _extract_boxed_answers_with_status(_strip_think_blocks(text))
    return answers


def normalize_math_answer(answer: str) -> str:
    """Normalize whitespace and simple LaTeX forms for exact matching."""

    value = str(answer).strip()
    boxed_answers, malformed = _extract_boxed_answers_with_status(value)
    if not malformed and len(boxed_answers) == 1:
        value = boxed_answers[0]

    value = value.strip()
    value = _strip_math_delimiters(value)
    value = value.replace(r"\left", "")
    value = value.replace(r"\right", "")
    value = value.replace(r"\dfrac", r"\frac")
    value = value.replace(r"\tfrac", r"\frac")
    for command in (r"\,", r"\!", r"\;", r"\:", r"\ "):
        value = value.replace(command, "")
    value = re.sub(r"\s+", "", value)
    value = _replace_latex_fracs(value)
    value = _strip_balanced_outer_braces(value)
    return value


def _extract_boxed_answers_with_status(text: str) -> Tuple[List[str], bool]:
    answers: List[str] = []
    malformed = False
    start = 0
    while True:
        boxed_index = text.find(_BOXED, start)
        if boxed_index < 0:
            break

        group_start = boxed_index + len(_BOXED)
        while group_start < len(text) and text[group_start].isspace():
            group_start += 1
        if group_start >= len(text) or text[group_start] != "{":
            malformed = True
            break

        inner_start = group_start + 1
        index = inner_start
        depth = 1
        while index < len(text):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    answers.append(text[inner_start:index])
                    break
            index += 1

        if depth != 0:
            malformed = True
            break
        start = index + 1

    return answers, malformed


def _strip_think_blocks(text: str) -> str:
    return re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)


def _has_unclosed_think_block(text: str) -> bool:
    depth = 0
    for match in re.finditer(r"</think>|<think\b[^>]*>", text, flags=re.IGNORECASE):
        tag = match.group(0).lower()
        if tag.startswith("</"):
            if depth > 0:
                depth -= 1
        else:
            depth += 1
    return depth > 0


def _boxed_is_final_only(text: str) -> bool:
    boxed_index = text.find(_BOXED)
    if boxed_index < 0:
        return False
    close_index = _boxed_close_index(text, boxed_index)
    if close_index is None:
        return False

    suffix = text[close_index + 1 :].strip()
    return _is_allowed_final_suffix(suffix)


def _boxed_close_index(text: str, boxed_index: int) -> Optional[int]:
    group_start = boxed_index + len(_BOXED)
    while group_start < len(text) and text[group_start].isspace():
        group_start += 1
    if group_start >= len(text) or text[group_start] != "{":
        return None

    depth = 1
    index = group_start + 1
    while index < len(text):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def _is_allowed_final_suffix(suffix: str) -> bool:
    if not suffix:
        return True
    return re.fullmatch(r"(?:\$|\\\)|\\\]|[\s.。!！])*", suffix) is not None


def _strip_math_delimiters(value: str) -> str:
    pairs = (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]"))
    changed = True
    while changed:
        changed = False
        stripped = value.strip()
        for prefix, suffix in pairs:
            if stripped.startswith(prefix) and stripped.endswith(suffix):
                value = stripped[len(prefix) : len(stripped) - len(suffix)]
                changed = True
                break
    return value


def _replace_latex_fracs(value: str) -> str:
    grouped = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    compact = re.compile(r"\\frac([+-]?\d+)([+-]?\d+)")
    previous = None
    while previous != value:
        previous = value
        value = grouped.sub(r"\1/\2", value)
        value = compact.sub(r"\1/\2", value)
    return value


def _strip_balanced_outer_braces(value: str) -> str:
    while value.startswith("{") and value.endswith("}"):
        depth = 0
        balanced_outer = True
        for index, char in enumerate(value):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and index != len(value) - 1:
                    balanced_outer = False
                    break
        if balanced_outer and depth == 0:
            value = value[1:-1]
        else:
            break
    return value


def _symbolically_equivalent(left: str, right: str) -> bool:
    left_value = _safe_fraction_eval(left)
    right_value = _safe_fraction_eval(right)
    return left_value is not None and right_value is not None and left_value == right_value


def _safe_fraction_eval(expression: str) -> Optional[Fraction]:
    if len(expression) > _MAX_SYMBOLIC_EXPR_CHARS:
        return None
    if not re.fullmatch(r"[0-9+\-*/().^]+", expression):
        return None

    python_expression = expression.replace("^", "**")
    try:
        tree = ast.parse(python_expression, mode="eval")
    except SyntaxError:
        return None

    if sum(1 for _ in ast.walk(tree)) > _MAX_SYMBOLIC_AST_NODES:
        return None
    try:
        value = _eval_fraction_node(tree.body)
    except (ArithmeticError, ValueError, TypeError):
        return None
    if not _fraction_is_bounded(value):
        return None
    return value


def _eval_fraction_node(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("unsupported constant")
        return _bounded_fraction(Fraction(str(node.value)))
    if isinstance(node, ast.Num):  # pragma: no cover - compatibility for old ASTs
        return _bounded_fraction(Fraction(str(node.n)))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_fraction_node(node.operand)
        if isinstance(node.op, ast.USub):
            value = -value
        return _bounded_fraction(value)
    if isinstance(node, ast.BinOp):
        left = _eval_fraction_node(node.left)
        right = _eval_fraction_node(node.right)
        if isinstance(node.op, ast.Add):
            return _bounded_fraction(left + right)
        if isinstance(node.op, ast.Sub):
            return _bounded_fraction(left - right)
        if isinstance(node.op, ast.Mult):
            return _bounded_fraction(left * right)
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ArithmeticError("division by zero")
            return _bounded_fraction(left / right)
        if isinstance(node.op, ast.Pow):
            if right.denominator != 1 or abs(right.numerator) > 8:
                raise ValueError("unsupported exponent")
            return _bounded_fraction(left ** right.numerator)
    raise ValueError("unsupported expression")


def _bounded_fraction(value: Fraction) -> Fraction:
    if not _fraction_is_bounded(value):
        raise ValueError("fraction too large")
    return value


def _fraction_is_bounded(value: Fraction) -> bool:
    return (
        abs(value.numerator) <= _MAX_FRACTION_COMPONENT
        and abs(value.denominator) <= _MAX_FRACTION_COMPONENT
    )
