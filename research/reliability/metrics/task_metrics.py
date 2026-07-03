"""Task consequence metrics for fault-injected predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClassificationFailure:
    critical_failure: bool
    severity: float
    baseline_class: int
    faulted_class: int
    confidence_delta: float


def classification_failure(
    baseline_logits: np.ndarray,
    faulted_logits: np.ndarray,
    target_class: int,
    *,
    confidence_threshold: float = 0.20,
) -> ClassificationFailure:
    baseline_prob = _softmax(baseline_logits)
    baseline_class = int(np.argmax(baseline_prob))
    baseline_correct = baseline_class == target_class
    if not np.all(np.isfinite(faulted_logits)):
        return ClassificationFailure(
            critical_failure=baseline_correct,
            severity=1.0 if baseline_correct else 0.0,
            baseline_class=baseline_class,
            faulted_class=-1,
            confidence_delta=1.0 if baseline_correct else 0.0,
        )
    faulted_prob = _softmax(faulted_logits)
    faulted_class = int(np.argmax(faulted_prob))
    changed_to_wrong = baseline_correct and faulted_class != target_class
    confidence_delta = float(abs(faulted_prob[target_class] - baseline_prob[target_class]))
    critical = changed_to_wrong or (baseline_correct and confidence_delta >= confidence_threshold)
    severity = float(max(changed_to_wrong, confidence_delta))
    return ClassificationFailure(critical, severity, baseline_class, faulted_class, confidence_delta)


@dataclass(frozen=True)
class DetectionFailure:
    critical_failure: bool
    severity: float
    missed_targets: int
    false_positives: int
    class_changes: int
    localization_degradation: float
    baseline_true_positives: int = 0


def detection_failure(
    baseline: np.ndarray,
    faulted: np.ndarray,
    *,
    confidence_threshold: float = 0.25,
    match_iou: float = 0.50,
) -> DetectionFailure:
    """Compare `[x1,y1,x2,y2,score,class]` detections after confidence filtering."""
    base = _filter_detections(baseline, confidence_threshold)
    failed = _filter_detections(faulted, confidence_threshold)
    used: set[int] = set()
    missed = 0
    class_changes = 0
    localization_loss: list[float] = []
    for reference in base:
        best_index, best_iou = _best_match(reference[:4], failed, used)
        if best_index is None or best_iou < match_iou:
            missed += 1
            continue
        used.add(best_index)
        if int(reference[5]) != int(failed[best_index, 5]):
            class_changes += 1
        localization_loss.append(1.0 - best_iou)
    false_positives = max(0, len(failed) - len(used))
    localization = float(np.mean(localization_loss)) if localization_loss else float(missed > 0)
    severity = _normalised_detection_severity(missed, false_positives, class_changes, localization, len(base))
    return DetectionFailure(
        critical_failure=bool(missed or false_positives or class_changes or localization > (1.0 - match_iou)),
        severity=severity,
        missed_targets=missed,
        false_positives=false_positives,
        class_changes=class_changes,
        localization_degradation=localization,
    )


def target_aware_detection_failure(
    baseline: np.ndarray,
    faulted: np.ndarray,
    targets: np.ndarray,
    *,
    confidence_threshold: float = 0.25,
    match_iou: float = 0.50,
) -> DetectionFailure:
    base = _filter_detections(baseline, confidence_threshold)
    failed = _filter_detections(faulted, confidence_threshold)
    ground_truth = np.asarray(targets, dtype=np.float32).reshape((-1, 5))
    references = _baseline_true_positives(base, ground_truth, match_iou)
    matched_failed: set[int] = set()
    missed = 0
    class_changes = 0
    localization_loss: list[float] = []
    for reference in references:
        best_index, best_iou = _best_match(reference[:4], failed, matched_failed)
        if best_index is None or best_iou < match_iou:
            missed += 1
            continue
        matched_failed.add(best_index)
        if int(reference[5]) != int(failed[best_index, 5]):
            class_changes += 1
        localization_loss.append(1.0 - best_iou)

    matched_to_baseline: set[int] = set()
    for reference in base:
        best_index, best_iou = _best_match(reference[:4], failed, matched_to_baseline)
        if best_index is not None and best_iou >= match_iou:
            matched_to_baseline.add(best_index)
    new_false_positives = max(0, len(failed) - len(matched_to_baseline))
    localization = float(np.mean(localization_loss)) if localization_loss else float(missed > 0)
    severity = _normalised_detection_severity(
        missed,
        new_false_positives,
        class_changes,
        localization,
        len(references),
    )
    critical = bool(missed or new_false_positives or class_changes or localization > (1.0 - match_iou))
    return DetectionFailure(
        critical_failure=critical,
        severity=severity,
        missed_targets=missed,
        false_positives=new_false_positives,
        class_changes=class_changes,
        localization_degradation=localization,
        baseline_true_positives=int(len(references)),
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("baseline logits must be finite")
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def _filter_detections(detections: np.ndarray, threshold: float) -> np.ndarray:
    values = np.asarray(detections, dtype=np.float32).reshape((-1, 6))
    return values[values[:, 4] >= threshold]


def _best_match(box: np.ndarray, detections: np.ndarray, used: set[int]) -> tuple[int | None, float]:
    best_index = None
    best_iou = 0.0
    for index, candidate in enumerate(detections):
        if index in used:
            continue
        overlap = _iou(box, candidate[:4])
        if overlap > best_iou:
            best_index, best_iou = index, overlap
    return best_index, best_iou


def _baseline_true_positives(baseline: np.ndarray, targets: np.ndarray, match_iou: float) -> np.ndarray:
    used: set[int] = set()
    true_positives = []
    for detection in baseline:
        best_index = None
        best_iou = 0.0
        for index, target in enumerate(targets):
            if index in used or int(detection[5]) != int(target[4]):
                continue
            overlap = _iou(detection[:4], target[:4])
            if overlap > best_iou:
                best_index, best_iou = index, overlap
        if best_index is not None and best_iou >= match_iou:
            used.add(best_index)
            true_positives.append(detection)
    return np.asarray(true_positives, dtype=np.float32).reshape((-1, 6))


def _normalised_detection_severity(
    missed: int,
    false_positives: int,
    class_changes: int,
    localization: float,
    protected_detection_count: int,
) -> float:
    consequence_count = missed + false_positives + class_changes
    denominator = max(1, protected_detection_count)
    return float(min(1.0, consequence_count / denominator + localization))


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    x1, y1 = np.maximum(left[:2], right[:2])
    x2, y2 = np.minimum(left[2:], right[2:])
    intersection = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    left_area = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    right_area = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0
