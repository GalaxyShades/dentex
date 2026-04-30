"""Experiment definitions for the local DENTEX research pipelines."""

from __future__ import annotations

from core.paths import MODELS_ROOT

BASELINE_EXPERIMENTS = [
    {
        "name": "baseline_yolo11n_1024",
        "pathology_model": "yolo11n.pt",
        "tooth_model": "yolo11n.pt",
        "imgsz": 1024,
        "epochs": 100,
        "mosaic": 0.1,
        "scale": 0.15,
        "degrees": 8.0,
        "enable_tta": False,
        "assignment_mode": "hungarian",
    },
    {
        "name": "larger_pathology_yolo11s",
        "pathology_model": "yolo11s.pt",
        "tooth_model": "yolo11n.pt",
        "imgsz": 1024,
        "epochs": 100,
        "mosaic": 0.1,
        "scale": 0.15,
        "degrees": 8.0,
        "enable_tta": False,
        "assignment_mode": "hungarian",
    },
    {
        "name": "higher_resolution_1280",
        "pathology_model": "yolo11n.pt",
        "tooth_model": "yolo11n.pt",
        "imgsz": 1280,
        "epochs": 100,
        "mosaic": 0.1,
        "scale": 0.15,
        "degrees": 8.0,
        "enable_tta": False,
        "assignment_mode": "hungarian",
    },
    {
        "name": "low_augmentation",
        "pathology_model": "yolo11n.pt",
        "tooth_model": "yolo11n.pt",
        "imgsz": 1024,
        "epochs": 100,
        "mosaic": 0.0,
        "scale": 0.05,
        "degrees": 3.0,
        "enable_tta": False,
        "assignment_mode": "hungarian",
    },
    {
        "name": "baseline_with_tta",
        "pathology_model": "yolo11n.pt",
        "tooth_model": "yolo11n.pt",
        "imgsz": 1024,
        "epochs": 100,
        "mosaic": 0.1,
        "scale": 0.15,
        "degrees": 8.0,
        "enable_tta": True,
        "assignment_mode": "hungarian",
    },
]

ASSIGNMENT_ABLATIONS = ["nearest", "iou_first", "hungarian"]

BASE_REGULARISATION_EXPERIMENT = {
    "name": "base_config",
    "pathology_model": "yolo11n.pt",
    "tooth_model": "yolo11n.pt",
    "imgsz": 1024,
    "epochs": 100,
    "patience": 20,
    "lr0": 0.01,
    "weight_decay": 0.0005,
    "mosaic": 0.1,
    "scale": 0.15,
    "degrees": 8.0,
    "enable_tta": True,
    "assignment_mode": "nearest",
    "pathology_conf": 0.15,
    "tooth_conf": 0.25,
    "predict_iou": 0.50,
}

REGULARISATION_EXPERIMENTS = [
    dict(BASE_REGULARISATION_EXPERIMENT),
    dict(BASE_REGULARISATION_EXPERIMENT, name="shorter_patience", patience=8),
    dict(BASE_REGULARISATION_EXPERIMENT, name="lower_lr", lr0=0.003),
    dict(
        BASE_REGULARISATION_EXPERIMENT,
        name="higher_weight_decay",
        weight_decay=0.002,
    ),
    dict(
        BASE_REGULARISATION_EXPERIMENT,
        name="stricter_prediction_confidence",
        pathology_conf=0.25,
        tooth_conf=0.35,
    ),
]

BEST_AB_CONFIG = {
    "name": "best_ab_yolo11n_1024_tta",
    "pathology_model": "yolo11n.pt",
    "tooth_model": "yolo11n.pt",
    "imgsz": 1024,
    "epochs": 100,
    "mosaic": 0.1,
    "scale": 0.15,
    "degrees": 8.0,
    "enable_tta": True,
    "assignment_mode": "nearest",
}

QUADRANT_EXPERIMENTS = [
    {
        "name": "quadrant_yolo11n_1024_standard",
        "quadrant_model": "yolo11n.pt",
        "imgsz": 1024,
        "epochs": 100,
        "mosaic": 0.1,
        "scale": 0.15,
        "degrees": 8.0,
    },
    {
        "name": "quadrant_yolo11n_1280_standard",
        "quadrant_model": "yolo11n.pt",
        "imgsz": 1280,
        "epochs": 100,
        "mosaic": 0.1,
        "scale": 0.15,
        "degrees": 8.0,
    },
    {
        "name": "quadrant_yolo11s_1024_standard",
        "quadrant_model": "yolo11s.pt",
        "imgsz": 1024,
        "epochs": 100,
        "mosaic": 0.1,
        "scale": 0.15,
        "degrees": 8.0,
    },
    {
        "name": "quadrant_yolo11n_1024_low_aug",
        "quadrant_model": "yolo11n.pt",
        "imgsz": 1024,
        "epochs": 100,
        "mosaic": 0.0,
        "scale": 0.05,
        "degrees": 3.0,
    },
]

QUADRANT_ASSIGNMENT_MODES = ["nearest", "iou_first", "hungarian"]
QUADRANT_TTA_MODES = [False, True]

FINAL_CONFIG = {
    "name": "final_model_c_lower_lr_yolo11s_quadrant",
    "pathology_model": "yolo11n.pt",
    "tooth_model": "yolo11n.pt",
    "quadrant_model": "yolo11s.pt",
    "imgsz": 1024,
    "epochs": 500,
    "patience": 30,
    "lr0": 0.003,
    "weight_decay": 0.0005,
    "mosaic": 0.1,
    "scale": 0.15,
    "degrees": 8.0,
    "ab_enable_tta": True,
    "quadrant_enable_tta": False,
    "tooth_assignment_mode": "nearest",
    "quadrant_assignment_mode": "nearest",
}

BEST_MODEL_WEIGHTS = {
    "pathology": MODELS_ROOT / "pathology_best.pt",
    "tooth": MODELS_ROOT / "tooth_best.pt",
    "quadrant": MODELS_ROOT / "quadrant_best.pt",
}
