"""Three-class patient/recording metrics, with explicit undefined ROC AUCs.

This module uses only the standard library. AUC is the positive/negative
ranking probability, counting a tied pair as one half. No model is loaded.
"""

from __future__ import annotations

from collections import Counter
import math


LABELS = ("NM", "AD", "MCI")


def binary_auc(targets, scores) -> float | None:
    """Return binary ROC AUC, or None when either class is absent."""

    pairs = sorted(zip(scores, targets), key=lambda pair: pair[0])
    positives = sum(bool(target) for _, target in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return None
    positive_rank_sum = 0.0
    start = 0
    while start < len(pairs):
        end = start + 1
        while end < len(pairs) and pairs[end][0] == pairs[start][0]:
            end += 1
        mean_rank = (start + 1 + end) / 2
        positive_rank_sum += mean_rank * sum(bool(target) for _, target in pairs[start:end])
        start = end
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def classification_metrics(targets, probs) -> dict:
    """Calculate NM=0 / AD=1 / MCI=2 metrics from class probabilities.

    String class names are also accepted. Macro F1 includes all three declared
    classes. Balanced accuracy averages recall over classes present in targets.
    A class AUC is undefined without both positives and negatives; the complete
    three-class macro AUC is null if any class AUC is undefined.
    """

    raw_targets, raw_probabilities = list(targets), list(probs)
    if not raw_targets or len(raw_targets) != len(raw_probabilities):
        raise ValueError("Targets and probabilities must have the same nonzero row count.")
    labels = []
    for value in raw_targets:
        if isinstance(value, str) and value in LABELS:
            labels.append(LABELS.index(value))
            continue
        try:
            label = int(value)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("Targets must be NM/AD/MCI or integer class indices 0, 1, 2.") from exc
        if isinstance(value, bool) or label not in range(3) or value != label:
            raise ValueError("Targets must be NM/AD/MCI or integer class indices 0, 1, 2.")
        labels.append(label)
    probabilities = []
    for row in raw_probabilities:
        try:
            values = [float(value) for value in row]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Each probability row must contain three finite numeric values.") from exc
        if len(values) != 3 or any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("Each probability row must contain three finite values in [0, 1].")
        if not math.isclose(sum(values), 1, rel_tol=0, abs_tol=1e-6):
            raise ValueError("Class probabilities must sum to one within 1e-6.")
        probabilities.append(values)
    confusion = [[0] * 3 for _ in LABELS]
    for target, probability in zip(labels, probabilities):
        predicted = max(range(3), key=lambda index: probability[index])
        confusion[target][predicted] += 1
    support = Counter(labels)
    per_class = {}
    for index, label in enumerate(LABELS):
        true_positive = confusion[index][index]
        predicted_count = sum(row[index] for row in confusion)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support[index] if support[index] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        auc = binary_auc([target == index for target in labels], [row[index] for row in probabilities])
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1,
                            "auc": auc, "support": support[index]}
    aucs = [values["auc"] for values in per_class.values()]
    return {
        "count": len(labels),
        "accuracy": sum(confusion[index][index] for index in range(3)) / len(labels),
        "balanced_accuracy": sum(per_class[LABELS[index]]["recall"] for index in support) / len(support),
        "macro_f1": sum(values["f1"] for values in per_class.values()) / 3,
        "loss": -sum(math.log(max(1e-12, min(row[target], 1)))
                     for target, row in zip(labels, probabilities)) / len(labels),
        "macro_ovr_auc": sum(aucs) / 3 if all(value is not None for value in aucs) else None,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "confusion_matrix_labels": list(LABELS),
    }
