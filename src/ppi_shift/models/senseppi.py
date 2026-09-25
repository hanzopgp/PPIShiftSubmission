"""Clean-init SENSE-PPI architecture contracts.

The public SENSE-PPI implementation is a Siamese ESM2-residue encoder followed by
a three-layer bidirectional GRU and a small binary head.  This module contains the
architecture-only adapter used for compatibility/symmetry tests.  Cluster scoring
may use the already audited ExternalPPI standardized runtime, but must not load the
published downstream SENSE checkpoint.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

try:
    import torch
    from torch import Tensor, nn

    _TORCH_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as error:  # Metadata-only imports remain torch-free.
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]

    class _NNPlaceholder:
        Module = object

    nn = _NNPlaceholder()  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = error

MODEL_ID = "senseppi"
MODEL_VERSION = "ppi_shift.senseppi_clean_init.v1"


class SensePPI(nn.Module):
    """Siamese SENSE-PPI head over residue-level ESM representations."""

    def __init__(
        self,
        input_dimension: int = 640,
        *,
        hidden_size: int = 128,
        layers: int = 3,
        dropout: float = 0.5,
    ) -> None:
        if _TORCH_IMPORT_ERROR is not None:
            raise ModuleNotFoundError(
                "SENSE-PPI tensor execution requires PyTorch; architecture metadata is available"
            ) from _TORCH_IMPORT_ERROR
        super().__init__()
        self.input_dimension = input_dimension
        self.hidden_size = hidden_size
        self.layers = layers
        self.encoder = nn.GRU(
            input_dimension,
            hidden_size,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def encode(self, sequences: Sequence[Tensor]) -> Tensor:
        if not sequences:
            raise ValueError("SENSE-PPI requires at least one sequence")
        lengths = torch.tensor(
            [sequence.shape[0] for sequence in sequences], device=sequences[0].device
        )
        padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        packed = nn.utils.rnn.pack_padded_sequence(
            padded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _output, hidden = self.encoder(packed)
        return hidden[-2:].transpose(0, 1).reshape(len(sequences), self.hidden_size * 2)

    def logits(self, left: Sequence[Tensor], right: Sequence[Tensor]) -> Tensor:
        if len(left) != len(right):
            raise ValueError("SENSE-PPI pair batches must be aligned")
        left_encoded = self.encode(left)
        right_encoded = self.encode(right)
        return self.head(left_encoded * right_encoded).flatten()

    def forward(self, left: Sequence[Tensor], right: Sequence[Tensor]) -> Tensor:
        return torch.sigmoid(self.logits(left, right))


def architecture_metadata(
    input_dimension: int = 640,
    *,
    representation_model: str | None = None,
    representation_revision: str | None = None,
) -> dict[str, object]:
    resolved_model = representation_model or (
        "facebook/esm2_t36_3B_UR50D" if input_dimension == 2560 else "facebook/esm2_t30_150M_UR50D"
    )
    return {
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "official_architecture": "Siamese ESM2 residue representations + 3-layer bidirectional GRU",
        "representation_model": resolved_model,
        "representation_revision": representation_revision or "required_artifact_revision",
        "input_dimension": input_dimension,
        "hidden_size": 128,
        "layers": 3,
        "dropout": 0.5,
        "pair_operator": "elementwise_product",
        "downstream_checkpoint": "forbidden; clean initialization only",
    }
