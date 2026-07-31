"""
Schema loading and validation.

This reuses the exact resume_schema.json produced in Part 4, so the API
and the offline evaluation script are validating against the same
contract instead of two schemas quietly drifting apart.
"""

import json
from pathlib import Path

from jsonschema import Draft7Validator

from app.config import settings


class SchemaValidationError(Exception):
    """Raised when a model output fails schema validation."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def _load_schema() -> dict:
    schema_path = Path(settings.schema_path)
    if not schema_path.is_absolute():
        # resolve relative to project root, not cwd, so it works regardless
        # of where uvicorn is launched from
        schema_path = Path(__file__).resolve().parent.parent / settings.schema_path
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


_schema = _load_schema()
_validator = Draft7Validator(_schema)


def validate_resume_output(data: dict) -> None:
    """
    Validate a parsed resume dict against resume_schema.json.

    Raises SchemaValidationError with a list of human-readable messages
    if validation fails. Returns None (silently) if valid.
    """
    errors = sorted(_validator.iter_errors(data), key=lambda e: e.path)
    if errors:
        messages = [
            f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}"
            for e in errors
        ]
        raise SchemaValidationError(messages)
