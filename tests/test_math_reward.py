import json
from pathlib import Path

import pytest

from posttrain_lab.rewards.math_reward import (
    MathRewardConfig,
    extract_boxed_answers,
    math_boxed_v001,
    normalize_math_answer,
    score_math_boxed_v001,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "rewards" / "math_boxed_v001_cases.jsonl"


def _fixture_cases():
    return [json.loads(line) for line in FIXTURE_PATH.read_text().splitlines() if line.strip()]


@pytest.mark.parametrize("case", _fixture_cases(), ids=lambda case: case["id"])
def test_math_boxed_v001_fixture_cases(case):
    config = MathRewardConfig(
        allow_symbolic_equivalence=case.get("allow_symbolic_equivalence", False),
        max_output_chars=case.get("max_output_chars", 20_000),
    )

    result = score_math_boxed_v001(case["completion"], case["answer"], config=config)

    assert result.score == case["expected_score"]
    assert result.reason == case["expected_reason"]


def test_extract_boxed_answers_handles_nested_latex_groups():
    assert extract_boxed_answers(r"Final: \boxed{\frac{1}{2}}") == [r"\frac{1}{2}"]


def test_normalize_math_answer_removes_simple_latex_noise():
    assert normalize_math_answer(r" \left( \frac{ 1 } { 2 } \right) ") == "(1/2)"


def test_math_boxed_v001_returns_float_score():
    assert math_boxed_v001(r"\boxed{4}", "4") == 1.0
    assert math_boxed_v001(r"\boxed{5}", "4") == 0.0


def test_malformed_box_before_valid_answer_is_rejected():
    result = score_math_boxed_v001(r"Bad \boxed{4 then final \boxed{4}", "4")

    assert result.score == 0.0
    assert result.reason == "malformed_boxed_answer"


def test_math_boxed_v001_strips_closed_think_blocks_before_parsing():
    result = score_math_boxed_v001("<think>Try 3. Wrong: \\boxed{3}</think>\n\\boxed{4}", "4")

    assert result.score == 1.0
    assert result.reason == "exact_match"
    assert result.normalized_prediction == "4"
    assert extract_boxed_answers("<think>Wrong: \\boxed{3}</think>\n\\boxed{4}") == ["4"]


def test_math_boxed_v001_accepts_reasoning_before_single_final_boxed_answer():
    completion = (
        "Natalia sold 48 clips in April. In May she sold half as many, "
        "so 48 + 24 = 72. $\\boxed{72}$"
    )

    result = score_math_boxed_v001(completion, "72")

    assert result.score == 1.0
    assert result.reason == "exact_match"
    assert result.normalized_prediction == "72"


def test_math_boxed_v001_strips_think_then_accepts_visible_final_boxed_answer():
    completion = "<think>Wrong hidden attempt: \\boxed{71}</think>\nThe final answer is $\\boxed{72}$."

    result = score_math_boxed_v001(completion, "72")

    assert result.score == 1.0
    assert result.reason == "exact_match"
    assert result.normalized_prediction == "72"


def test_math_boxed_v001_does_not_accept_unclosed_think_blocks():
    result = score_math_boxed_v001("<think>Try 4.\n\\boxed{4}", "4")

    assert result.score == 0.0
