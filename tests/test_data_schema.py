import os
import subprocess
import sys
from pathlib import Path

from posttrain_lab.data.validate import validate_jsonl


FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[1]


def cli_env():
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def error_text(dataset_type, fixture_name):
    errors = validate_jsonl(dataset_type, FIXTURES / fixture_name)
    return "\n".join(str(error) for error in errors)


def test_good_sft_fixture_passes():
    assert validate_jsonl("sft", FIXTURES / "sft_good.jsonl") == []


def test_good_rlvr_fixture_passes():
    assert validate_jsonl("rlvr", FIXTURES / "rlvr_good.jsonl") == []


def test_invalid_json_reports_path_and_line():
    text = error_text("sft", "bad_invalid_json.jsonl")

    assert "bad_invalid_json.jsonl:1" in text
    assert "invalid JSON" in text


def test_missing_required_field_reports_line():
    text = error_text("sft", "bad_missing_field.jsonl")

    assert "bad_missing_field.jsonl:1" in text
    assert "missing required field: metadata" in text


def test_duplicate_ids_report_original_and_duplicate_lines():
    text = error_text("sft", "bad_duplicate_ids.jsonl")

    assert "bad_duplicate_ids.jsonl:2" in text
    assert "duplicate id: duplicate-id" in text
    assert "first seen on line 1" in text


def test_invalid_split_reports_line():
    text = error_text("sft", "bad_invalid_split.jsonl")

    assert "bad_invalid_split.jsonl:1" in text
    assert "invalid split: dev" in text


def test_invalid_message_role_reports_line():
    text = error_text("sft", "bad_invalid_role.jsonl")

    assert "bad_invalid_role.jsonl:1" in text
    assert "invalid messages[0].role: developer" in text


def test_empty_content_reports_line():
    text = error_text("sft", "bad_empty_content.jsonl")

    assert "bad_empty_content.jsonl:1" in text
    assert "messages[0].content must be non-empty" in text


def test_rlvr_prompt_rejects_assistant_messages():
    text = error_text("rlvr", "bad_rlvr_assistant_prompt.jsonl")

    assert "bad_rlvr_assistant_prompt.jsonl:1" in text
    assert "RLVR prompt[1].role must be user, got assistant" in text


def test_cli_passes_for_good_fixture():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "posttrain_lab.data.validate",
            "--type",
            "sft",
            "--path",
            str(FIXTURES / "sft_good.jsonl"),
        ],
        text=True,
        capture_output=True,
        check=False,
        env=cli_env(),
    )

    assert result.returncode == 0
    assert "valid" in result.stdout


def test_cli_fails_for_bad_fixture_with_line_number():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "posttrain_lab.data.validate",
            "--type",
            "rlvr",
            "--path",
            str(FIXTURES / "bad_rlvr_assistant_prompt.jsonl"),
        ],
        text=True,
        capture_output=True,
        check=False,
        env=cli_env(),
    )

    assert result.returncode == 1
    assert "bad_rlvr_assistant_prompt.jsonl:1" in result.stderr
    assert "RLVR prompt[1].role must be user" in result.stderr
