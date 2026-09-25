from __future__ import annotations

import json

import pytest

from scripts.run_e003_reused_e001_scores import _checkpoint_path


def test_reuse_checkpoint_path_uses_run_manifest_fallback(tmp_path) -> None:
    run_path = tmp_path / "run"
    run_path.mkdir()
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text("{}\n", encoding="utf-8")
    (run_path / "run_manifest.json").write_text(
        json.dumps({"checkpoint": {"path": str(checkpoint_path)}}), encoding="utf-8"
    )

    assert _checkpoint_path(run_path) == checkpoint_path


def test_sense_toy_forward_handles_variable_residue_lengths_and_is_symmetric() -> None:
    torch = pytest.importorskip("torch")
    from ppi_shift.models.senseppi import SensePPI, architecture_metadata

    torch.manual_seed(1729)
    model = SensePPI(input_dimension=4, hidden_size=2, layers=3)
    model.eval()
    left = [torch.randn(3, 4), torch.randn(2, 4)]
    right = [torch.randn(4, 4), torch.randn(2, 4)]

    logits = model.logits(left, right)
    swapped = model.logits(right, left)

    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
    assert torch.allclose(logits, swapped)
    assert architecture_metadata(input_dimension=2560)["pair_operator"] == "elementwise_product"


def test_rapppid_awd_toy_forward_uses_faithful_c3_path() -> None:
    torch = pytest.importorskip("torch")
    from ppi_shift.models.rapppid import RapppidAWD, awd_architecture_metadata

    torch.manual_seed(1729)
    model = RapppidAWD(
        250,
        embedding_size=8,
        recurrent_layers=2,
        bi_reduce="last",
        class_head_name="concat",
    )
    model.eval()
    tokens = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long)

    logits = model.logits(tokens, tokens.flip(1))

    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
    metadata = awd_architecture_metadata()
    assert metadata["optimizer"] == "Ranger21"
    assert metadata["averaged_weights"] == "stochastic_weight_averaging"
    assert metadata["class_head"].startswith("official_manuscript_C3_concat")
    assert model.class_head_name == "concat"
    assert model.class_head[0].in_features == 16
    assert model.class_head[0].out_features == 4
