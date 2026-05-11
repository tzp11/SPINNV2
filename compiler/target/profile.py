"""Target profile loading and lightweight validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL_KEYS = {
    "name",
    "word_size",
    "endianness",
    "alignment",
    "memory",
    "features",
    "layouts",
    "backends",
    "ops",
}


def bundled_profiles_dir() -> Path:
    return Path(__file__).resolve().parent / "profiles"


def load_target_profile(name_or_path: str) -> dict[str, Any]:
    path = Path(name_or_path)
    if not path.exists():
        path = bundled_profiles_dir() / f"{name_or_path}.json"

    if not path.exists():
        raise FileNotFoundError(f"target profile not found: {name_or_path}")

    with path.open("r", encoding="utf-8") as fp:
        profile = json.load(fp)

    validate_target_profile(profile)
    return profile


def validate_target_profile(profile: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_TOP_LEVEL_KEYS - set(profile))
    if missing:
        raise ValueError(f"target profile missing keys: {', '.join(missing)}")

    if profile["endianness"] not in {"little", "big"}:
        raise ValueError("target profile endianness must be 'little' or 'big'")

    memory = profile["memory"]
    for key in ("activation_arena_max", "scratch_arena_max", "allow_runtime_malloc"):
        if key not in memory:
            raise ValueError(f"target profile memory missing key: {key}")

    if not isinstance(profile["ops"], dict):
        raise ValueError("target profile ops must be an object")

