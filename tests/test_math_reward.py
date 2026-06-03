import json
import importlib.util
import sys
import types
from pathlib import Path

import pytest

from posttrain_lab.rewards.math_reward import (
    MathRewardConfig,
    extract_boxed_answers,
    math_boxed_v001,
    math_boxed_verl_v001,
    normalize_math_answer,
    score_math_boxed_verl_v001,
    score_math_boxed_v001,
)
import posttrain_lab.rewards.math_reward as math_reward


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


def test_verl_compute_score_accepts_keyword_signature_and_list_ground_truth():
    result = math_reward.compute_score(
        data_source="olympiad_bench_boxed",
        solution_str=r"Final answer: \boxed{2}",
        ground_truth=["2"],
        extra_info={"index": 0},
    )

    assert result["score"] == 1.0
    assert result["reason"] == "exact_match"
    assert result["reward_version"] == "math_boxed_v001"
    assert result["ground_truth_index"] == 0
    assert result["data_source"] == "olympiad_bench_boxed"


def test_verl_compute_score_rejects_multiple_boxed_answers():
    result = math_reward.compute_score(
        r"Check \boxed{4}. Final \boxed{4}",
        "4",
    )

    assert result["score"] == 0.0
    assert result["reason"] == "multiple_boxed_answers"


def test_verl_reward_wrapper_loads_from_file_path_without_sys_modules_registration():
    wrapper_path = Path(__file__).parents[1] / "src" / "posttrain_lab" / "rewards" / "verl_math_reward.py"
    spec = importlib.util.spec_from_file_location("verl_external_reward", wrapper_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    result = module.compute_score(
        data_source="verl-smoke",
        solution_str=r"Final answer: \boxed{4}",
        ground_truth="4",
        extra_info={},
    )
    assert result["score"] == 1.0
    assert result["reason"] == "exact_match"


def test_verl_reward_wrapper_sanitizes_none_metadata_for_validation_metrics():
    wrapper_path = Path(__file__).parents[1] / "src" / "posttrain_lab" / "rewards" / "verl_math_reward.py"
    spec = importlib.util.spec_from_file_location("verl_external_reward", wrapper_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    result = module.compute_score(
        data_source=None,
        solution_str="No final answer.",
        ground_truth="4",
        extra_info={},
    )

    assert result["score"] == 0.0
    assert result["reason"] == "no_boxed_answer"
    assert result["extracted_answer"] == ""
    assert result["normalized_prediction"] == ""
    assert result["normalized_answer"] == ""
    assert result["data_source"] == ""
    assert result["ground_truth_index"] == 0


def test_math_boxed_verl_v001_accepts_correct_boxed_answer_with_trailing_text():
    result = score_math_boxed_verl_v001(
        r"The answer is \boxed{4}. This follows from addition.",
        "4",
    )

    assert result.score == 1.0
    assert result.reason == "exact_match"
    assert result.version == "math_boxed_verl_v001"
    assert result.normalized_prediction == "4"


def test_math_boxed_verl_v001_gives_format_credit_for_wrong_boxed_answer():
    result = score_math_boxed_verl_v001(r"The answer is \boxed{5}.", "4")

    assert result.score == 0.1
    assert result.reason == "format_correct_answer_mismatch"
    assert result.normalized_prediction == "5"
    assert math_boxed_verl_v001(r"The answer is \boxed{5}.", "4") == 0.1


def test_math_boxed_verl_v001_uses_last_boxed_answer_like_common_final_answer_extractors():
    result = score_math_boxed_verl_v001(r"Try \boxed{3}. Final answer: \boxed{4}.", "4")

    assert result.score == 1.0
    assert result.reason == "exact_match"
    assert result.normalized_prediction == "4"


def test_verl_reward_wrapper_exposes_common_verl_style_reward():
    wrapper_path = Path(__file__).parents[1] / "src" / "posttrain_lab" / "rewards" / "verl_math_reward.py"
    spec = importlib.util.spec_from_file_location("verl_external_reward", wrapper_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    result = module.compute_score_verl_style(
        data_source="verl-smoke",
        solution_str=r"The final answer is \boxed{5}.",
        ground_truth="4",
        extra_info={},
    )

    assert result["score"] == 0.1
    assert result["reason"] == "format_correct_answer_mismatch"
    assert result["reward_version"] == "math_boxed_verl_v001"


def test_sympy_engine_uses_latex2sympy2_extended_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "latex2sympy2", None)

    fake_extended = types.ModuleType("latex2sympy2_extended")

    def fake_latex2sympy(expression):
        import sympy

        return sympy.sympify(expression.replace("^", "**"))

    fake_extended.latex2sympy = fake_latex2sympy
    monkeypatch.setitem(sys.modules, "latex2sympy2_extended", fake_extended)

    result = score_math_boxed_v001(
        r"\boxed{2*x+2}",
        "2*(x+1)",
        config=MathRewardConfig(allow_symbolic_equivalence=True, symbolic_equivalence_engine="sympy"),
    )

    assert result.score == 1.0
    assert result.reason == "sympy_equivalence"


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


def test_sympy_equivalence_is_disabled_by_default_for_algebra():
    result = score_math_boxed_v001(r"\boxed{2(x+1)}", "2x+2")

    assert result.score == 0.0
    assert result.reason == "answer_mismatch"


def test_sympy_equivalence_accepts_decimal_fraction_when_enabled():
    pytest.importorskip("latex2sympy2")

    result = score_math_boxed_v001(
        r"\boxed{0.5}",
        r"\frac{1}{2}",
        config=MathRewardConfig(allow_symbolic_equivalence=True, symbolic_equivalence_engine="sympy"),
    )

    assert result.score == 1.0
    assert result.reason == "sympy_equivalence"


def test_sympy_equivalence_accepts_algebraic_expansion_when_enabled():
    pytest.importorskip("latex2sympy2")

    result = score_math_boxed_v001(
        r"\boxed{2(x+1)}",
        "2x+2",
        config=MathRewardConfig(allow_symbolic_equivalence=True, symbolic_equivalence_engine="sympy"),
    )

    assert result.score == 1.0
    assert result.reason == "sympy_equivalence"


def test_sympy_equivalence_accepts_unordered_sets_when_enabled():
    pytest.importorskip("latex2sympy2")

    result = score_math_boxed_v001(
        r"\boxed{\{2,1,3\}}",
        r"\{1,2,3\}",
        config=MathRewardConfig(allow_symbolic_equivalence=True, symbolic_equivalence_engine="sympy"),
    )

    assert result.score == 1.0
    assert result.reason == "sympy_equivalence"


def test_sympy_equivalence_rejects_wrong_algebra_when_enabled():
    pytest.importorskip("latex2sympy2")

    result = score_math_boxed_v001(
        r"\boxed{2(x+1)}",
        "2x+3",
        config=MathRewardConfig(allow_symbolic_equivalence=True, symbolic_equivalence_engine="sympy"),
    )

    assert result.score == 0.0
    assert result.reason == "answer_mismatch"


def test_sympy_equivalence_rejects_equation_parser_hacking():
    pytest.importorskip("latex2sympy2")

    result = score_math_boxed_v001(
        r"\boxed{x=1}",
        "1",
        config=MathRewardConfig(allow_symbolic_equivalence=True, symbolic_equivalence_engine="sympy"),
    )

    assert result.score == 0.0
    assert result.reason == "answer_mismatch"
