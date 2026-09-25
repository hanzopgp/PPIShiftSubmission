"""Load the versioned pre-freeze source configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def config_path(config: dict[str, Any], *keys: str) -> Path:
    value: Any = config
    for key in keys:
        value = value[key]
    return Path(str(value))
