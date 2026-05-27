"""Strict JSONL schema definitions for SFT and RLVR data."""

from dataclasses import dataclass
from pathlib import Path

VALID_SPLITS = {"train", "val", "test"}
SFT_MESSAGE_ROLES = {"system", "user", "assistant"}

SFT_REQUIRED_FIELDS = {"id", "split", "messages", "metadata"}
RLVR_REQUIRED_FIELDS = {"id", "split", "prompt", "verifier", "metadata"}
MESSAGE_REQUIRED_FIELDS = {"role", "content"}
METADATA_REQUIRED_FIELDS = {"source", "domain", "difficulty", "license"}
VERIFIER_REQUIRED_FIELDS = {"type", "answer"}


@dataclass(frozen=True)
class ValidationError:
    """A schema validation error tied to a JSONL file and line."""

    path: Path
    line_number: int
    message: str

    @property
    def line(self):
        """Line-number alias used by external test suites."""

        return self.line_number

    def __str__(self):
        return f"{self.path}:{self.line_number}: {self.message}"


@dataclass(frozen=True)
class ValidationReport:
    """Validation result for one JSONL file."""

    ok: bool
    num_rows: int
    errors: list
