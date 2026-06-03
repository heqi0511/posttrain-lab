"""Deterministic boxed-answer math reward."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any, List, Optional, Tuple


REWARD_VERSION = "math_boxed_v001"
VERL_STYLE_REWARD_VERSION = "math_boxed_verl_v001"
VERL_STYLE_FORMAT_REWARD = 0.1
SUPPORTED_REWARD_VERSIONS = frozenset({REWARD_VERSION, VERL_STYLE_REWARD_VERSION})
_BOXED = r"\boxed"
_MAX_SYMBOLIC_EXPR_CHARS = 120
_MAX_SYMBOLIC_AST_NODES = 64
_MAX_FRACTION_COMPONENT = 10**12


@dataclass(frozen=True)
class MathRewardConfig:
    """Configuration for math_boxed_v001."""

    allow_symbolic_equivalence: bool = False
    symbolic_equivalence_engine: str = "fraction"
    max_output_chars: int = 20_000
    max_answer_chars: int = 1_000
    max_symbolic_expr_chars: int = _MAX_SYMBOLIC_EXPR_CHARS
    max_symbolic_ast_nodes: int = _MAX_SYMBOLIC_AST_NODES
    max_symbolic_collection_size: int = 32


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
        config = replace(base, allow_symbolic_equivalence=allow_symbolic_equivalence)
    return score_math_boxed_v001(completion, answer, config=config).score


def math_boxed_verl_v001(
    completion: str,
    answer: str,
    *,
    config: Optional[MathRewardConfig] = None,
    allow_symbolic_equivalence: Optional[bool] = None,
    format_reward: float = VERL_STYLE_FORMAT_REWARD,
) -> float:
    """Return the scalar common-verl-style boxed math reward."""

    if allow_symbolic_equivalence is not None:
        base = config or MathRewardConfig()
        config = replace(base, allow_symbolic_equivalence=allow_symbolic_equivalence)
    return score_math_boxed_verl_v001(
        completion,
        answer,
        config=config,
        format_reward=format_reward,
    ).score


def score_math_boxed_by_version(
    completion: str,
    answer: str,
    *,
    reward_version: str = REWARD_VERSION,
    config: Optional[MathRewardConfig] = None,
) -> MathRewardResult:
    """Score a boxed math completion using an explicit reward version."""

    if reward_version == REWARD_VERSION:
        return score_math_boxed_v001(completion, answer, config=config)
    if reward_version == VERL_STYLE_REWARD_VERSION:
        return score_math_boxed_verl_v001(completion, answer, config=config)
    raise ValueError(f"unsupported reward_version: {reward_version}")


def compute_score(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """verl-compatible adapter for ``math_boxed_v001``.

    Supported call forms:
    - ``compute_score(solution_str, ground_truth)``
    - ``compute_score(data_source=..., solution_str=..., ground_truth=..., extra_info=...)``
    - ``compute_score(data_source, solution_str, ground_truth, extra_info)``

    The adapter unwraps dataset-provided reference lists, but the completion
    parser and reward semantics remain exactly those of ``math_boxed_v001``.
    """

    solution_str, ground_truth, data_source, _, config = _parse_compute_score_args(args, kwargs)
    candidates = _ground_truth_candidates(ground_truth)
    if not candidates:
        return {
            "score": 0.0,
            "reason": "missing_ground_truth",
            "reward_version": REWARD_VERSION,
            "extracted_answer": None,
            "normalized_prediction": None,
            "normalized_answer": None,
            "ground_truth_index": None,
            "num_ground_truths": 0,
            "data_source": data_source,
        }

    best_index = 0
    best_result: Optional[MathRewardResult] = None
    for index, candidate in enumerate(candidates):
        result = score_math_boxed_v001(solution_str, candidate, config=config)
        if best_result is None or result.score > best_result.score:
            best_index = index
            best_result = result
        if result.score == 1.0:
            break

    assert best_result is not None
    return {
        "score": best_result.score,
        "reason": best_result.reason,
        "reward_version": best_result.version,
        "extracted_answer": best_result.extracted_answer,
        "normalized_prediction": best_result.normalized_prediction,
        "normalized_answer": best_result.normalized_answer,
        "ground_truth_index": best_index,
        "num_ground_truths": len(candidates),
        "data_source": data_source,
    }


def compute_score_verl_style(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """verl-compatible adapter for the common-verl-style boxed math reward."""

    format_reward = float(kwargs.pop("format_reward", VERL_STYLE_FORMAT_REWARD))
    solution_str, ground_truth, data_source, _, config = _parse_compute_score_args(args, kwargs)
    candidates = _ground_truth_candidates(ground_truth)
    if not candidates:
        return {
            "score": 0.0,
            "reason": "missing_ground_truth",
            "reward_version": VERL_STYLE_REWARD_VERSION,
            "extracted_answer": None,
            "normalized_prediction": None,
            "normalized_answer": None,
            "ground_truth_index": None,
            "num_ground_truths": 0,
            "data_source": data_source,
        }

    best_index = 0
    best_result: Optional[MathRewardResult] = None
    for index, candidate in enumerate(candidates):
        result = score_math_boxed_verl_v001(
            solution_str,
            candidate,
            config=config,
            format_reward=format_reward,
        )
        if best_result is None or result.score > best_result.score:
            best_index = index
            best_result = result
        if result.score == 1.0:
            break

    assert best_result is not None
    return {
        "score": best_result.score,
        "reason": best_result.reason,
        "reward_version": best_result.version,
        "extracted_answer": best_result.extracted_answer,
        "normalized_prediction": best_result.normalized_prediction,
        "normalized_answer": best_result.normalized_answer,
        "ground_truth_index": best_index,
        "num_ground_truths": len(candidates),
        "data_source": data_source,
    }


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


def _parse_compute_score_args(
    args: Tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Tuple[str, Any, Optional[str], Any, MathRewardConfig]:
    remaining = dict(kwargs)
    config = remaining.pop("config", None) or MathRewardConfig()
    data_source = remaining.pop("data_source", None)
    solution_str = remaining.pop("solution_str", None)
    ground_truth = remaining.pop("ground_truth", None)
    extra_info = remaining.pop("extra_info", None)

    if len(args) == 2:
        if solution_str is not None or ground_truth is not None:
            raise TypeError("compute_score received duplicate solution_str or ground_truth")
        solution_str, ground_truth = args
    elif len(args) == 4:
        has_duplicate = (
            data_source is not None
            or solution_str is not None
            or ground_truth is not None
            or extra_info is not None
        )
        if has_duplicate:
            raise TypeError("compute_score received duplicate verl reward arguments")
        data_source, solution_str, ground_truth, extra_info = args
    elif args:
        raise TypeError("compute_score expects either 2 positional args or verl keyword args")

    if solution_str is None:
        solution_str = ""
    if data_source is not None:
        data_source = str(data_source)
    if not isinstance(config, MathRewardConfig):
        raise TypeError("config must be a MathRewardConfig")
    return str(solution_str), ground_truth, data_source, extra_info, config


def _ground_truth_candidates(ground_truth: Any) -> List[str]:
    if isinstance(ground_truth, dict) and "ground_truth" in ground_truth:
        return _ground_truth_candidates(ground_truth["ground_truth"])
    if ground_truth is None:
        return []
    if isinstance(ground_truth, str):
        value = ground_truth.strip()
        return [value] if value else []
    if isinstance(ground_truth, (list, tuple, set)):
        candidates: List[str] = []
        for item in ground_truth:
            candidates.extend(_ground_truth_candidates(item))
        return candidates
    value = str(ground_truth).strip()
    return [value] if value else []


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

    equivalence_reason = _symbolic_equivalence_reason(normalized_prediction, normalized_answer, config)
    if equivalence_reason:
        return MathRewardResult(
            score=1.0,
            reason=equivalence_reason,
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


def score_math_boxed_verl_v001(
    completion: str,
    answer: str,
    *,
    config: Optional[MathRewardConfig] = None,
    format_reward: float = VERL_STYLE_FORMAT_REWARD,
) -> MathRewardResult:
    """Score using a common verl-style boxed final-answer verifier.

    This variant is intentionally more permissive than ``math_boxed_v001``:
    it extracts the last well-formed visible boxed answer, gives full credit
    when it matches the reference, and gives a small format reward when the
    answer is boxed but wrong.
    """

    config = config or MathRewardConfig()
    if len(completion) > config.max_output_chars:
        return MathRewardResult(score=0.0, reason="output_too_long", version=VERL_STYLE_REWARD_VERSION)
    if len(answer) > config.max_answer_chars:
        return MathRewardResult(score=0.0, reason="answer_too_long", version=VERL_STYLE_REWARD_VERSION)
    if _has_unclosed_think_block(completion):
        return MathRewardResult(score=0.0, reason="unclosed_think_block", version=VERL_STYLE_REWARD_VERSION)

    scored_completion = _strip_think_blocks(completion)
    boxed_answers, malformed = _extract_boxed_answers_with_status(scored_completion)
    if malformed:
        return MathRewardResult(score=0.0, reason="malformed_boxed_answer", version=VERL_STYLE_REWARD_VERSION)
    if not boxed_answers:
        return MathRewardResult(score=0.0, reason="no_boxed_answer", version=VERL_STYLE_REWARD_VERSION)

    extracted_answer = boxed_answers[-1]
    normalized_prediction = normalize_math_answer(extracted_answer)
    if not normalized_prediction:
        return MathRewardResult(
            score=0.0,
            reason="empty_boxed_answer",
            version=VERL_STYLE_REWARD_VERSION,
            extracted_answer=extracted_answer,
            normalized_prediction=normalized_prediction,
            normalized_answer=normalize_math_answer(answer),
        )

    normalized_answer = normalize_math_answer(answer)
    if normalized_prediction == normalized_answer:
        return MathRewardResult(
            score=1.0,
            reason="exact_match",
            version=VERL_STYLE_REWARD_VERSION,
            extracted_answer=extracted_answer,
            normalized_prediction=normalized_prediction,
            normalized_answer=normalized_answer,
        )

    equivalence_reason = _symbolic_equivalence_reason(normalized_prediction, normalized_answer, config)
    if equivalence_reason:
        return MathRewardResult(
            score=1.0,
            reason=equivalence_reason,
            version=VERL_STYLE_REWARD_VERSION,
            extracted_answer=extracted_answer,
            normalized_prediction=normalized_prediction,
            normalized_answer=normalized_answer,
        )

    return MathRewardResult(
        score=max(0.0, min(1.0, float(format_reward))),
        reason="format_correct_answer_mismatch",
        version=VERL_STYLE_REWARD_VERSION,
        extracted_answer=extracted_answer,
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


def _symbolic_equivalence_reason(left: str, right: str, config: MathRewardConfig) -> Optional[str]:
    if not config.allow_symbolic_equivalence:
        return None

    engine = str(config.symbolic_equivalence_engine or "fraction").lower()
    if engine in {"none", "off", "disabled"}:
        return None
    if engine in {"fraction", "safe_fraction"}:
        return "symbolic_equivalence" if _symbolically_equivalent(left, right) else None
    if engine in {"sympy", "latex2sympy2"}:
        return "sympy_equivalence" if _sympy_equivalent(left, right, config) else None
    return None


def _sympy_equivalent(left: str, right: str, config: MathRewardConfig) -> bool:
    left_obj = _parse_latex_sympy_object(left, config)
    right_obj = _parse_latex_sympy_object(right, config)
    if left_obj is None or right_obj is None:
        return False
    return _sympy_objects_equivalent(left_obj, right_obj, config)


def _parse_latex_sympy_object(expression: str, config: MathRewardConfig) -> Optional[Any]:
    if len(expression) > int(config.max_symbolic_expr_chars):
        return None
    if _contains_sympy_rejected_syntax(expression):
        return None
    latex2sympy = _load_latex2sympy()
    if latex2sympy is None:
        return None
    try:
        parsed = latex2sympy(expression)
    except Exception:
        return None
    if not _sympy_object_is_bounded(parsed, config):
        return None
    return parsed


def _load_latex2sympy() -> Optional[Any]:
    for module_name in ("latex2sympy2_extended", "latex2sympy2"):
        try:
            module = __import__(module_name, fromlist=["latex2sympy"])
        except ImportError:
            continue
        latex2sympy = getattr(module, "latex2sympy", None)
        if latex2sympy is not None:
            return latex2sympy
    return None


def _contains_sympy_rejected_syntax(expression: str) -> bool:
    if "=" in expression:
        return True
    if any(token in expression for token in (r"\text", r"\mathrm", r"\begin", r"\cases")):
        return True
    if re.search(r"\b(?:yes|no|true|false|or|and)\b", expression, flags=re.IGNORECASE):
        return True
    if not re.fullmatch(r"[0-9A-Za-z\\{}()[\]+\-*/^_,.|\s]+", expression):
        return True
    return False


def _sympy_object_is_bounded(value: Any, config: MathRewardConfig) -> bool:
    sequence = _as_sequence(value)
    if sequence is not None:
        if len(sequence) > int(config.max_symbolic_collection_size):
            return False
        return all(_sympy_object_is_bounded(item, config) for item in sequence)

    try:
        import sympy
    except ImportError:
        return False
    if not isinstance(value, sympy.Basic):
        return False
    return _sympy_node_count(value) <= int(config.max_symbolic_ast_nodes)


def _sympy_objects_equivalent(left: Any, right: Any, config: MathRewardConfig) -> bool:
    left_sequence = _as_sequence(left)
    right_sequence = _as_sequence(right)
    if left_sequence is not None or right_sequence is not None:
        if left_sequence is None or right_sequence is None:
            return False
        if len(left_sequence) != len(right_sequence):
            return False
        if len(left_sequence) > int(config.max_symbolic_collection_size):
            return False
        unmatched = list(right_sequence)
        for left_item in left_sequence:
            match_index = next(
                (
                    index
                    for index, right_item in enumerate(unmatched)
                    if _sympy_expr_equivalent(left_item, right_item, config)
                ),
                None,
            )
            if match_index is None:
                return False
            unmatched.pop(match_index)
        return not unmatched
    return _sympy_expr_equivalent(left, right, config)


def _sympy_expr_equivalent(left: Any, right: Any, config: MathRewardConfig) -> bool:
    try:
        import sympy
    except ImportError:
        return False
    if not isinstance(left, sympy.Basic) or not isinstance(right, sympy.Basic):
        return False
    if (
        _sympy_node_count(left) > int(config.max_symbolic_ast_nodes)
        or _sympy_node_count(right) > int(config.max_symbolic_ast_nodes)
    ):
        return False
    try:
        difference = sympy.simplify(left - right)
    except Exception:
        return False
    if _sympy_node_count(difference) > int(config.max_symbolic_ast_nodes):
        return False
    if difference == 0:
        return True
    try:
        return difference.equals(0) is True
    except Exception:
        return False


def _sympy_node_count(value: Any) -> int:
    try:
        import sympy
    except ImportError:
        return _MAX_SYMBOLIC_AST_NODES + 1
    return sum(1 for _ in sympy.preorder_traversal(value))


def _as_sequence(value: Any) -> Optional[List[Any]]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return None


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
