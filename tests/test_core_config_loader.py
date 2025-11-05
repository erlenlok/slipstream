from __future__ import annotations

import json
from pathlib import Path

import pytest

from slipstream.core.config import (
    ConfigNotFoundError,
    load_layered_config,
)


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data))


def test_load_layered_config_merges_strategy_and_env(tmp_path):
    config_dir = tmp_path
    _write_json(config_dir / "global.json", {"a": 1, "nested": {"x": 1}})
    _write_json(config_dir / "gradient_live.json", {"b": 2, "nested": {"y": 2}})

    env_dir = config_dir / "environments" / "prod"
    env_dir.mkdir(parents=True)
    _write_json(env_dir / "gradient_live.json", {"nested": {"x": 10}, "c": 3})

    result = load_layered_config(
        "gradient_live",
        config_dir=config_dir,
        env="prod",
        overrides={"nested": {"z": 4}},
    )

    assert result["a"] == 1
    assert result["b"] == 2
    # Environment override wins for nested.x
    assert result["nested"]["x"] == 10
    # Strategy value persists
    assert result["nested"]["y"] == 2
    # Override adds new key
    assert result["nested"]["z"] == 4
    # Environment addition respected
    assert result["c"] == 3


def test_missing_strategy_config_raises(tmp_path):
    with pytest.raises(ConfigNotFoundError):
        load_layered_config("gradient_live", config_dir=tmp_path)

