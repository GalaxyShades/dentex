"""Shared configuration builders for DENTEX research pipelines."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from core.paths import DATA_ROOT, RESULTS_ROOT

SEED = 42

BASE_TRAINING_CONFIG = {
    "is_local": True,
    "dataset_source": "local data/dentex directory",
    "dataset_root": DATA_ROOT,
    "pathology_class_names": [],
    "enumeration_class_names": [f"tooth_{i}" for i in range(1, 9)],
    "imgsz": 1024,
    "epochs": 100,
    "sanity_mode": False,
    "sanity_epochs": 1,
    "sanity_fraction": 0.1,
    "sanity_imgsz": 640,
    "sanity_batch": 4,
    "sanity_max_eval_images": 24,
    "sanity_max_test_images": 40,
    "batch": 8,
    "patience": 20,
    "cache": False,
    "optimizer": "AdamW",
    "lr0": 0.01,
    "mosaic": 0.1,
    "mixup": 0.0,
    "degrees": 8.0,
    "translate": 0.03,
    "scale": 0.15,
    "hsv_h": 0.01,
    "hsv_s": 0.15,
    "hsv_v": 0.15,
    "fliplr": 0.0,
    "flipud": 0.0,
    "predict_iou": 0.5,
    "tta_fusion_iou": 0.55,
    "prune_heavy_outputs": True,
}


def local_device():
    return 0 if torch.cuda.is_available() else "cpu"


def local_workers() -> int:
    return min(8, max(2, (os.cpu_count() or 4) // 2))


def build_pipeline_config(
    project_dirname: str,
    overrides: dict | None = None,
) -> dict:
    cfg = dict(BASE_TRAINING_CONFIG)
    cfg.update(
        {
            "work_root": RESULTS_ROOT / project_dirname,
            "yolo_root": RESULTS_ROOT / project_dirname / "yolo",
            "runs_root": RESULTS_ROOT / project_dirname / "runs",
            "pred_root": RESULTS_ROOT / project_dirname / "predictions",
            "official_eval_root": RESULTS_ROOT / project_dirname / "official_eval",
            "device": local_device(),
            "workers": local_workers(),
        }
    )
    if overrides:
        cfg.update(overrides)
    return cfg


def apply_sanity_config(
    cfg: dict,
    experiments: list[dict],
    prediction_confidence_keys: tuple[str, ...] = (),
) -> list[dict]:
    if not cfg["sanity_mode"]:
        return experiments

    cfg["epochs"] = cfg["sanity_epochs"]
    cfg["patience"] = 1
    cfg["imgsz"] = cfg["sanity_imgsz"]
    cfg["batch"] = cfg["sanity_batch"]
    cfg["predict_iou"] = 0.4

    for key in prediction_confidence_keys:
        cfg[key] = 0.05

    return [
        dict(
            experiments[0],
            imgsz=cfg["sanity_imgsz"],
            epochs=cfg["sanity_epochs"],
            patience=1,
            enable_tta=False,
        )
    ]


def ensure_output_dirs(cfg: dict) -> None:
    for key in [
        "work_root",
        "yolo_root",
        "runs_root",
        "pred_root",
        "official_eval_root",
    ]:
        Path(cfg[key]).mkdir(parents=True, exist_ok=True)
