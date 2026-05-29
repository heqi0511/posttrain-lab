from posttrain_lab.data.math_sft_curation import curate_pair


def test_curate_pair_rejects_boolean_target():
    decision = curate_pair("Is the statement true?", r"\boxed{Yes}")

    assert decision.action == "reject"
    assert decision.reason == "boolean_target"


def test_curate_pair_rejects_logical_multiple_answer():
    decision = curate_pair("Find x.", r"\boxed{x=1 \text{ or } x=2}")

    assert decision.action == "reject"
    assert decision.reason == "logical_word_target"


def test_curate_pair_sanitizes_fraction_target():
    decision = curate_pair("Find the value.", r"\boxed{$\dfrac{\sqrt{2}}{2}$}")

    assert decision.action == "keep"
    assert decision.clean_target == r"\boxed{\frac{\sqrt{2}}{2}}"


def test_curate_pair_removes_degree_unit():
    decision = curate_pair("Find the angle in degrees.", r"\boxed{124.806^\circ}")

    assert decision.action == "keep"
    assert decision.clean_target == r"\boxed{124.806}"


def test_curate_pair_rejects_answer_leak_in_problem():
    decision = curate_pair("Find x. Answer. 9.", r"\boxed{9}")

    assert decision.action == "reject"
    assert decision.reason == "answer_leak_or_multipart_prompt"


def test_curate_pair_rejects_image_problem():
    decision = curate_pair("Use the figure below: ![](https://example.com/a.png)", r"\boxed{5}")

    assert decision.action == "reject"
    assert decision.reason == "answer_leak_or_multipart_prompt"


def test_curate_pair_rejects_depicted_below_problem():
    decision = curate_pair("A rectangle is folded as depicted below. Find the ratio.", r"\boxed{\sqrt{5}}")

    assert decision.action == "reject"
    assert decision.reason == "answer_leak_or_multipart_prompt"


def test_curate_pair_rejects_explanation_prompt():
    decision = curate_pair("Find n. Please explain your answer.", r"\boxed{7}")

    assert decision.action == "reject"
    assert decision.reason == "proof_or_boolean_prompt"


def test_curate_pair_standardizes_comma_list_as_set():
    decision = curate_pair("List all values.", r"\boxed{1,2,3}")

    assert decision.action == "keep"
    assert decision.clean_target == r"\boxed{\{1,2,3\}}"
