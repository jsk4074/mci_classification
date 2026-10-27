"""Known probability rankings, degenerate classes, and strict score validation."""

import json
import math
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from s01_metrics import classification_metrics


def test_known_confusion_losses_and_tied_probability_auc():
    targets = [0, 0, 0, 1, 1, 2]
    probabilities = [[.6, .3, .1], [.4, .5, .1], [.2, .7, .1],
                     [.4, .5, .1], [.1, .8, .1], [.15, .15, .7]]
    result = classification_metrics(targets, probabilities)
    assert result["confusion_matrix"] == [[1, 2, 0], [0, 2, 0], [0, 0, 1]]
    assert result["confusion_matrix_labels"] == ["NM", "AD", "MCI"]
    assert result["count"] == 6
    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["balanced_accuracy"] == pytest.approx(7 / 9)
    assert result["macro_f1"] == pytest.approx((.5 + 2 / 3 + 1) / 3)
    assert result["loss"] == pytest.approx(-sum(math.log(row[target]) for row, target in zip(probabilities, targets)) / 6)
    assert result["per_class"]["NM"]["auc"] == pytest.approx(5 / 6)
    assert result["per_class"]["AD"]["auc"] == pytest.approx(13 / 16)
    assert result["per_class"]["MCI"]["auc"] == 1
    assert result["macro_ovr_auc"] == pytest.approx(127 / 144)
    assert result == classification_metrics(["NM", "NM", "NM", "AD", "AD", "MCI"], probabilities)


def test_missing_class_auc_is_null_while_observed_class_metrics_remain_defined():
    result = classification_metrics([0, 1], [[.8, .1, .1], [.1, .8, .1]])
    assert result["macro_ovr_auc"] is None
    assert result["per_class"]["MCI"] == {"precision": 0, "recall": 0, "f1": 0, "auc": None, "support": 0}
    assert result["per_class"]["NM"]["auc"] == 1
    assert result["balanced_accuracy"] == 1
    assert result["macro_f1"] == pytest.approx(2 / 3)
    assert "NaN" not in json.dumps(result, allow_nan=False)
    single_class = classification_metrics([0], [[1, 0, 0]])
    assert all(values["auc"] is None for values in single_class["per_class"].values())


@pytest.mark.parametrize("targets,probabilities", [
    ([], []), ([0], []), ([3], [[.8, .1, .1]]), ([1.5], [[.8, .1, .1]]),
    ([0], [[float("nan"), .1, .1]]), ([0], [[.8, .1]]),
    ([0], [[.8, .2, .2]]), ([0], [[1.2, -.1, -.1]]),
])
def test_invalid_metric_inputs_are_rejected(targets, probabilities):
    with pytest.raises(ValueError):
        classification_metrics(targets, probabilities)
