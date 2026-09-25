"""Model contracts and bounded smoke adapters.

Each main model exposes one conceptual interface while preserving author semantics
where those semantics affect the prediction.
"""

from ppi_shift.models.frozen_esm import FrozenESMPairMLP, smoke_run, symmetric_pair_features

try:  # The compact Windows test environment does not need the optional torch adapters.
    from ppi_shift.models.rapppid import Rapppid
    from ppi_shift.models.senseppi import SensePPI
except ModuleNotFoundError:  # pragma: no cover - exercised only without torch
    Rapppid = None  # type: ignore[assignment]
    SensePPI = None  # type: ignore[assignment]

__all__ = [
    "FrozenESMPairMLP",
    "Rapppid",
    "SensePPI",
    "smoke_run",
    "symmetric_pair_features",
]
