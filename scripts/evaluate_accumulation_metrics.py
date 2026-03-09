#!/usr/bin/env python3
"""
Evaluate a medical video anonymization pipeline on strict PHI safety and utility.

Input model (JSON-compatible):
[
  {
    "frame_id": 0,
    "ground_truth_entities": [{"label": "PER", "text": "Max", "bbox": [x1,y1,x2,y2]}],
    "predicted_entities": [{"label": "PER", "text": "Max", "bbox": [x1,y1,x2,y2]}],
    "blurred_regions": [[x1,y1,x2,y2], ...]
  },
  ...
]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


BBox = Tuple[int, int, int, int]


@dataclass
class EvalConfig:
    iou_threshold: float = 0.3
    coverage_threshold: float = 0.5
    frame_width: int = 1920
    frame_height: int = 1080
    circle_radius: int = 460


def _area(box: BBox) -> float:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _intersection(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x_left = max(ax1, bx1)
    y_top = max(ay1, by1)
    x_right = min(ax2, bx2)
    y_bottom = min(ay2, by2)
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    return float((x_right - x_left) * (y_bottom - y_top))


def iou(a: BBox, b: BBox) -> float:
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def coverage(gt_box: BBox, pred_box: BBox) -> float:
    inter = _intersection(gt_box, pred_box)
    gt_area = _area(gt_box)
    return inter / gt_area if gt_area > 0 else 0.0


def _entity_matches(gt: Dict, pred: Dict, cfg: EvalConfig) -> bool:
    if gt.get("label") != pred.get("label"):
        return False
    gt_box = tuple(gt["bbox"])
    pred_box = tuple(pred["bbox"])
    return (
        iou(gt_box, pred_box) >= cfg.iou_threshold
        or coverage(gt_box, pred_box) >= cfg.coverage_threshold
    )


def phi_leakage_rate(frames: List[Dict], cfg: EvalConfig) -> float:
    """
    Strict frame-level leakage:
    A frame leaks if it has at least one GT entity and ANY GT entity is not matched.
    """
    frames_with_phi = 0
    leaked_frames = 0
    for frame in frames:
        gt_entities = frame.get("ground_truth_entities", [])
        pred_entities = frame.get("predicted_entities", [])
        if not gt_entities:
            continue
        frames_with_phi += 1

        frame_leaks = False
        for gt in gt_entities:
            matched = any(_entity_matches(gt, pred, cfg) for pred in pred_entities)
            if not matched:
                frame_leaks = True
                break
        leaked_frames += int(frame_leaks)

    if frames_with_phi == 0:
        return 0.0
    return 100.0 * leaked_frames / frames_with_phi


def entity_recall(frames: List[Dict], cfg: EvalConfig) -> float:
    """
    Standard recall = TP / (TP + FN), using one-to-one greedy matching.
    """
    tp = 0
    fn = 0
    for frame in frames:
        gt_entities = frame.get("ground_truth_entities", [])
        pred_entities = frame.get("predicted_entities", [])
        used_pred = set()

        for gt in gt_entities:
            match_idx = None
            for idx, pred in enumerate(pred_entities):
                if idx in used_pred:
                    continue
                if _entity_matches(gt, pred, cfg):
                    match_idx = idx
                    break
            if match_idx is not None:
                tp += 1
                used_pred.add(match_idx)
            else:
                fn += 1

    denom = tp + fn
    return tp / denom if denom else 0.0


def viewport_preservation_ratio(frames: List[Dict], cfg: EvalConfig) -> float:
    """
    VPR = unblurred safe endoscopic circle area / total safe circle area.
    """
    h, w = cfg.frame_height, cfg.frame_width
    yy, xx = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= cfg.circle_radius**2
    safe_area = float(circle.sum())

    frame_ratios = []
    for frame in frames:
        blurred = np.zeros((h, w), dtype=bool)
        for box in frame.get("blurred_regions", []):
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            if x2 > x1 and y2 > y1:
                blurred[y1:y2, x1:x2] = True

        blurred_safe = float(np.logical_and(circle, blurred).sum())
        unblurred_safe = max(0.0, safe_area - blurred_safe)
        frame_ratios.append(unblurred_safe / safe_area if safe_area else 0.0)

    if not frame_ratios:
        return 0.0
    return 100.0 * float(np.mean(frame_ratios))


def evaluate(frames: List[Dict], cfg: EvalConfig) -> pd.DataFrame:
    leakage = phi_leakage_rate(frames, cfg)
    recall = entity_recall(frames, cfg)
    vpr = viewport_preservation_ratio(frames, cfg)

    rows = [
        {"Metric": "PHI Leakage Rate (%)", "Value": round(leakage, 2)},
        {"Metric": "Entity Recall (NER)", "Value": round(recall, 4)},
        {"Metric": "Viewport Preservation Ratio (VPR, %)", "Value": round(vpr, 2)},
    ]
    return pd.DataFrame(rows)


def sample_frames() -> List[Dict]:
    return [
        {
            "frame_id": 1,
            "ground_truth_entities": [
                {"label": "PER", "text": "Alice", "bbox": [80, 40, 220, 95]},
                {"label": "DOB", "text": "1970-01-01", "bbox": [240, 40, 430, 95]},
            ],
            "predicted_entities": [
                {"label": "PER", "text": "Alice", "bbox": [78, 38, 222, 96]},
                {"label": "DOB", "text": "1970-01-01", "bbox": [236, 36, 432, 97]},
            ],
            "blurred_regions": [[0, 0, 500, 120]],
        },
        {
            "frame_id": 2,
            "ground_truth_entities": [
                {"label": "PER", "text": "Alice", "bbox": [82, 41, 223, 96]},
            ],
            "predicted_entities": [],
            "blurred_regions": [[0, 0, 260, 110]],
        },
        {
            "frame_id": 3,
            "ground_truth_entities": [
                {"label": "CASE", "text": "A-9921", "bbox": [450, 40, 590, 92]},
            ],
            "predicted_entities": [
                {"label": "CASE", "text": "A-9921", "bbox": [448, 38, 592, 95]},
            ],
            "blurred_regions": [[430, 20, 620, 110]],
        },
        {
            "frame_id": 4,
            "ground_truth_entities": [],
            "predicted_entities": [],
            "blurred_regions": [[0, 0, 140, 80]],
        },
        {
            "frame_id": 5,
            "ground_truth_entities": [
                {"label": "PER", "text": "Dr. Smith", "bbox": [1200, 70, 1440, 130]},
            ],
            "predicted_entities": [
                {"label": "PER", "text": "Dr. Smith", "bbox": [1190, 60, 1455, 140]},
            ],
            "blurred_regions": [[1180, 50, 1460, 145], [860, 430, 980, 560]],
        },
    ]


def main() -> None:
    cfg = EvalConfig()
    frames = sample_frames()
    df = evaluate(frames, cfg)
    print("\nEvaluation Summary")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
