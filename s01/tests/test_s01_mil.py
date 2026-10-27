"""Synthetic patient-bag pooling checks; no pretrained encoder is loaded."""

from pathlib import Path
import sys

import pytest


torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from s01_mil import PatientMIL, collate_bags, patient_class_weights


def test_collate_pads_recordings_and_preserves_targets_and_patient_order():
    first = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    second = torch.ones(1, 4)
    batch = collate_bags([
        {"embeddings": first, "target": 2, "patient_id": "synthetic_first"},
        {"embeddings": second, "target": 0, "patient_id": "synthetic_second"},
    ])
    assert batch["embeddings"].shape == (2, 3, 4)
    assert batch["mask"].dtype == torch.bool
    assert batch["mask"].tolist() == [[True, True, True], [True, False, False]]
    assert batch["targets"].tolist() == [2, 0]
    assert batch["patient_ids"] == ["synthetic_first", "synthetic_second"]
    torch.testing.assert_close(batch["embeddings"][0], first)
    assert torch.count_nonzero(batch["embeddings"][1, 1:]) == 0


@pytest.mark.parametrize("pooling", ["mean", "attention"])
def test_masked_padding_cannot_change_logits_or_receive_attention(pooling):
    torch.manual_seed(4)
    model = PatientMIL(8, pooling=pooling, projection_size=5, dropout=0).eval()
    values = torch.randn(2, 3, 8)
    mask = torch.tensor([[True, True, True], [True, False, False]])
    logits, weights = model(values, mask, return_attention=True)
    changed = values.clone()
    changed[~mask] = float("nan")
    changed = torch.cat([changed, torch.full((2, 4, 8), float("inf"))], dim=1)
    expanded_mask = torch.cat([mask, torch.zeros((2, 4), dtype=torch.bool)], dim=1)
    changed_logits, changed_weights = model(changed, expanded_mask, return_attention=True)
    assert logits.shape == (2, 3)
    torch.testing.assert_close(logits, changed_logits)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))
    torch.testing.assert_close(changed_weights.sum(dim=1), torch.ones(2))
    assert torch.count_nonzero(changed_weights[~expanded_mask]) == 0
    if pooling == "mean":
        torch.testing.assert_close(weights[0], torch.full((3,), 1 / 3))
        assert weights[1].tolist() == [1, 0, 0]


@pytest.mark.parametrize("pooling", ["mean", "attention"])
def test_patient_prediction_is_recording_order_invariant(pooling):
    torch.manual_seed(9)
    model = PatientMIL(6, pooling=pooling, projection_size=4, dropout=0).eval()
    values = torch.randn(2, 5, 6)
    mask = torch.tensor([[True] * 5, [True, True, False, False, False]])
    permutation = torch.tensor([4, 1, 3, 0, 2])
    original = model(values, mask)
    permuted = model(values[:, permutation], mask[:, permutation])
    torch.testing.assert_close(original, permuted)


def test_attention_is_trainable_and_padding_has_no_embedding_gradient():
    torch.manual_seed(11)
    model = PatientMIL(6, pooling="attention", projection_size=4, dropout=0)
    embeddings = torch.randn(2, 3, 6, requires_grad=True)
    mask = torch.tensor([[True, True, True], [True, False, False]])
    logits = model(embeddings, mask)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2])).backward()
    assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad)
               for parameter in model.attention.parameters())
    assert torch.isfinite(embeddings.grad).all()
    assert torch.count_nonzero(embeddings.grad[~mask]) == 0
    assert torch.count_nonzero(embeddings.grad[mask]) > 0


def test_class_weights_count_patients_and_use_inverse_class_frequency():
    weights = patient_class_weights([0, 0, 0, 1, 1, 2])
    assert weights.dtype == torch.float32
    torch.testing.assert_close(weights, torch.tensor([2 / 3, 1, 2]))


@pytest.mark.parametrize("targets", [[], [0, 1], [0, 1, 3], [0, 1, -1],
                                      [0.0, 1.0, 2.0], [[0, 1, 2]], [True, False]])
def test_invalid_or_missing_training_classes_are_rejected(targets):
    with pytest.raises(ValueError):
        patient_class_weights(targets)


@pytest.mark.parametrize("pooling", ["mean", "attention"])
def test_all_padding_bag_and_invalid_masks_are_rejected(pooling):
    model = PatientMIL(4, pooling=pooling, projection_size=3, dropout=0)
    values = torch.randn(2, 3, 4)
    with pytest.raises(ValueError, match="at least one valid"):
        model(values, torch.tensor([[True, True, True], [False, False, False]]))
    with pytest.raises(ValueError, match="bool"):
        model(values, torch.ones(2, 3))
    with pytest.raises(ValueError, match="shape"):
        model(torch.ones(2, 3, 5), torch.ones(2, 3, dtype=torch.bool))
    values[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(values, torch.ones(2, 3, dtype=torch.bool))


@pytest.mark.parametrize("kwargs", [{"pooling": "unknown"}, {"hidden_size": 0},
                                     {"projection_size": -1}, {"attention_size": 0},
                                     {"dropout": 1}, {"dropout": float("nan")}])
def test_invalid_head_configuration_is_rejected(kwargs):
    options = {"hidden_size": 4, **kwargs}
    with pytest.raises(ValueError):
        PatientMIL(**options)


@pytest.mark.parametrize("batch", [[], [{}], [
    {"embeddings": torch.ones(0, 4), "target": 0, "patient_id": "synthetic"}], [
    {"embeddings": torch.ones(2, 4), "target": 3, "patient_id": "synthetic"}], [
    {"embeddings": torch.ones(2, 4), "target": 0, "patient_id": ""}], [
    {"embeddings": torch.ones(2, 4), "target": 0, "patient_id": "synthetic"},
    {"embeddings": torch.ones(2, 5), "target": 1, "patient_id": "other"}],
])
def test_invalid_bags_are_rejected(batch):
    with pytest.raises(ValueError):
        collate_bags(batch)
