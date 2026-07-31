"""
Configuration management.

All runtime settings are read from environment variables (via a .env file
in local dev), with sensible defaults so the service can still boot without
one. Nothing model-related should be hardcoded elsewhere in the app --
if you need a new tunable, add it here.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()  # no-op in prod if a real .env isn't present; fine either way


@dataclass(frozen=True)
class Settings:
    model_name: str
    device: str
    max_new_tokens: int
    temperature: float
    schema_path: str


def _get_bool_or_str(key: str, default: str) -> str:
    return os.getenv(key, default)


def load_settings() -> Settings:
    return Settings(
        model_name=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct"),
        device=os.getenv("DEVICE", "cpu"),
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "1024")),
        temperature=float(os.getenv("TEMPERATURE", "0.1")),
        schema_path=os.getenv("SCHEMA_PATH", "app/resume_schema.json"),
    )


settings = load_settings()
