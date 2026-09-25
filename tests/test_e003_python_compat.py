from __future__ import annotations

from pathlib import Path

import pytest

E003_RUNTIME_SCRIPTS = (
    "freeze_e003_protocol.py",
    "submit_e003_core_dag.py",
    "submit_e003_core_staged.py",
    "submit_e003_datasail_dag.py",
    "submit_e003_external_dag.py",
    "dispatch_e003_datasail_scores.py",
    "dispatch_e003_external_scores.py",
    "dispatch_e003_datasail_scores_staged.py",
    "dispatch_e003_external_scores_staged.py",
)


@pytest.mark.parametrize("script_name", E003_RUNTIME_SCRIPTS)
def test_e003_runtime_scripts_keep_python310_utc_compatibility(script_name: str) -> None:
    source = (Path(__file__).parents[1] / "scripts" / script_name).read_text(encoding="utf-8")

    assert "from datetime import UTC" not in source
    if "from datetime import" in source:
        assert 'UTC = getattr(timezone, "utc")' in source
