"""RAPPPID architectures.

``Rapppid`` is retained as the immutable E002 approximation. ``RapppidAWD`` is
the separate E003 adapter for the pinned upstream AWD-LSTM contract.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

MODEL_ID = "rapppid"
MODEL_VERSION = "ppi_shift.rapppid_clean_init.v1"


class Mish(nn.Module):
    def forward(self, value: Tensor) -> Tensor:
        return value * torch.tanh(nn.functional.softplus(value))


class WeightDrop(nn.Module):
    """DropConnect wrapper matching the pinned RAPPPID implementation."""

    def __init__(self, module: nn.Module, weight_names: tuple[str, ...], dropout: float) -> None:
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError("RAPPPID weight dropout must be in [0, 1)")
        self.module = module
        self.weight_names = weight_names
        self.dropout = dropout
        if isinstance(module, nn.RNNBase):
            module.flatten_parameters = lambda: None
        for name in weight_names:
            parameter = getattr(module, name)
            del module._parameters[name]
            module.register_parameter(f"{name}_raw", nn.Parameter(parameter.detach().clone()))

    def _setweights(self) -> None:
        for name in self.weight_names:
            raw = getattr(self.module, f"{name}_raw")
            dropped = functional.dropout(raw, p=self.dropout, training=self.training)
            # Recent PyTorch RNN implementations can restore the original
            # parameter entry while moving the module between devices.  A
            # plain dropped tensor cannot be assigned while that entry is
            # still registered as an ``nn.Parameter``.
            self.module._parameters.pop(name, None)
            setattr(self.module, name, dropped)

    def forward(self, *args: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        self._setweights()
        return self.module(*args)


class RapppidAWD(nn.Module):
    """Pinned RAPPPID C3-style AWD-LSTM twin encoder and concat head.

    The official model is intentionally not made symmetric: its concat head is
    order-sensitive. E003 therefore exposes direct logits and leaves any
    orientation sensitivity as a measured property of the run.
    """

    def __init__(
        self,
        vocabulary_size: int,
        *,
        embedding_size: int = 64,
        recurrent_layers: int = 2,
        embedding_dropout: float = 0.3,
        weight_dropout: float = 0.3,
        class_head_dropout: float = 0.2,
        truncation_length: int = 512,
        bi_reduce: str = "last",
        class_head_name: str = "concat",
    ) -> None:
        super().__init__()
        if vocabulary_size < 2:
            raise ValueError("RAPPPID vocabulary must include padding and one token")
        if recurrent_layers < 1 or embedding_size % 2:
            raise ValueError("RAPPPID requires positive layers and an even embedding size")
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.recurrent_layers = recurrent_layers
        self.embedding_dropout_rate = embedding_dropout
        self.weight_dropout_rate = weight_dropout
        self.class_head_dropout_rate = class_head_dropout
        self.truncation_length = truncation_length
        self.bi_reduce = bi_reduce
        self.class_head_name = class_head_name
        if bi_reduce not in {"concat", "last"}:
            raise ValueError("RAPPPID E003 supports the official concat or last reduction")
        if class_head_name not in {"concat", "mult"}:
            raise ValueError("RAPPPID E003 supports the official concat or mult head")
        self.embedding = nn.Embedding(vocabulary_size, embedding_size, padding_idx=0)
        recurrent_hidden = embedding_size // 2 if bi_reduce == "concat" else embedding_size
        recurrent = nn.LSTM(
            embedding_size,
            recurrent_hidden,
            recurrent_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.rnn = WeightDrop(recurrent, ("weight_hh_l0",), weight_dropout)
        encoded_dimension = embedding_size if bi_reduce == "concat" else recurrent_hidden
        self.fc = nn.Linear(encoded_dimension, embedding_size)
        self.activation = Mish()
        if class_head_name == "concat":
            self.class_head = nn.Sequential(
                nn.Linear(embedding_size * 2, embedding_size // 2),
                nn.Dropout(class_head_dropout),
                Mish(),
                nn.Linear(embedding_size // 2, 1),
            )
        else:
            self.class_head = nn.Sequential(
                WeightDrop(
                    nn.Linear(embedding_size, embedding_size // 2),
                    ("weight",),
                    class_head_dropout,
                ),
                Mish(),
                WeightDrop(nn.Linear(embedding_size // 2, 1), ("weight",), class_head_dropout),
            )

    def embedding_dropout(self, tokens: Tensor) -> Tensor:
        if not self.training or self.embedding_dropout_rate <= 0:
            weight = self.embedding.weight
        else:
            mask = self.embedding.weight.new_empty((self.vocabulary_size, 1))
            mask = functional.dropout(mask.fill_(1.0), p=self.embedding_dropout_rate, training=True)
            weight = mask * self.embedding.weight
        return functional.embedding(
            tokens,
            weight,
            padding_idx=self.embedding.padding_idx,
            max_norm=self.embedding.max_norm,
            norm_type=self.embedding.norm_type,
            scale_grad_by_freq=self.embedding.scale_grad_by_freq,
            sparse=self.embedding.sparse,
        )

    def encode(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2:
            raise ValueError("RAPPPID tokens must have shape (batch, length)")
        lengths = tokens.ne(0).sum(dim=1)
        if torch.any(lengths == 0):
            raise ValueError("RAPPPID cannot encode an all-padding sequence")
        max_length = min(int(lengths.max().item()), self.truncation_length)
        trimmed = tokens[:, :max_length]
        embedded = self.embedding_dropout(trimmed)
        _output, (hidden, _cell) = self.rnn(embedded)
        if self.bi_reduce == "concat":
            encoded = hidden[-2:].permute(1, 0, 2).reshape(tokens.shape[0], self.embedding_size)
        else:
            encoded = hidden[-1]
        return self.activation(self.fc(encoded))

    def logits(self, left: Tensor, right: Tensor) -> Tensor:
        if left.shape[0] != right.shape[0]:
            raise ValueError("RAPPPID pair batches must be aligned")
        left_encoded = self.encode(left)
        right_encoded = self.encode(right)
        if self.class_head_name == "mult":
            left_encoded = (left_encoded - left_encoded.mean()) / left_encoded.std()
            right_encoded = (right_encoded - right_encoded.mean()) / right_encoded.std()
            return self.class_head(self.activation(left_encoded * right_encoded)).flatten()
        return self.class_head(torch.cat((left_encoded, right_encoded), dim=1)).flatten()

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        return torch.sigmoid(self.logits(left, right))


def awd_architecture_metadata() -> dict[str, object]:
    return {
        "model_id": MODEL_ID,
        "model_version": "ppi_shift.rapppid_awd_clean_init.v1",
        "official_repository": "https://github.com/jszym/rapppid",
        "source_revision": "c3a28be56fb3bd96ccf8cc44ea9145cfcadeaf6c",
        "architecture": "modified_AWD_LSTM_twin_encoder",
        "tokenizer": "SentencePiece_unigram",
        "vocabulary_size": 250,
        "embedding_size": 64,
        "recurrent_layers": 2,
        "bidirectional_hidden_per_direction": 32,
        "embedding_dropout": 0.3,
        "weight_dropconnect": 0.3,
        "class_head_dropout": 0.2,
        "class_head": "official_manuscript_C3_concat_128_to_32_mish_to_1_dropout_0.2",
        "optimizer": "Ranger21",
        "averaged_weights": "stochastic_weight_averaging",
        "c3_hyperparameters": {
            "batch_size": 80,
            "epochs": 100,
            "learning_rate": 0.01,
            "weight_decay": 0.0001,
            "bi_reduce": "last",
            "class_head": "concat",
            "variational_dropout": False,
            "warmdown_start_pct": 0.72,
        },
        "truncation": 512,
        "pair_orientation": "official_direct_concat_order_sensitive",
        "downstream_checkpoint": "forbidden; clean initialization only",
    }


class Rapppid(nn.Module):
    """RAPPPID's embedding + bidirectional LSTM + concat head.

    ``bi_reduce=concat`` yields one representation of ``embedding_size`` per protein,
    matching the upstream constructor.  Pair scoring is made explicitly symmetric
    by averaging both input orientations in ``symmetric_logits``.
    """

    def __init__(
        self,
        vocabulary_size: int,
        *,
        embedding_size: int = 64,
        recurrent_layers: int = 2,
        lstm_dropout: float = 0.0,
        class_head_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if vocabulary_size < 2:
            raise ValueError("RAPPPID vocabulary must include padding and one token")
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocabulary_size, embedding_size, padding_idx=0)
        self.rnn = nn.LSTM(
            embedding_size,
            embedding_size // 2,
            recurrent_layers,
            dropout=lstm_dropout if recurrent_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(embedding_size, embedding_size)
        self.activation = Mish()
        self.class_head = nn.Sequential(
            nn.Linear(embedding_size * 2, embedding_size),
            nn.Dropout(class_head_dropout),
            Mish(),
            nn.Linear(embedding_size, 1),
        )

    def encode(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2:
            raise ValueError("RAPPPID tokens must have shape (batch, length)")
        embedded = self.embedding(tokens)
        _output, (hidden, _cell) = self.rnn(embedded)
        encoded = hidden[-2:].permute(1, 0, 2).reshape(tokens.shape[0], self.embedding_size)
        return self.activation(self.fc(encoded))

    def logits(self, left: Tensor, right: Tensor) -> Tensor:
        if left.shape[0] != right.shape[0]:
            raise ValueError("RAPPPID pair batches must be aligned")
        return self.class_head(torch.cat((self.encode(left), self.encode(right)), dim=1)).flatten()

    def symmetric_logits(self, left: Tensor, right: Tensor) -> Tensor:
        return 0.5 * (self.logits(left, right) + self.logits(right, left))

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        return torch.sigmoid(self.symmetric_logits(left, right))


def architecture_metadata() -> dict[str, object]:
    return {
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "official_architecture": "SentencePiece token embedding + 2-layer bidirectional LSTM",
        "vocabulary_size": 250,
        "embedding_size": 64,
        "recurrent_layers": 2,
        "bi_reduce": "concat",
        "class_head": "concat, two layers, Mish",
        "truncation": 512,
        "sentencepiece_sampling": False,
        "downstream_checkpoint": "forbidden; clean initialization only",
    }
