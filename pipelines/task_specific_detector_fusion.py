"""Local pipeline converted from the project notebook.

This script is intended to be run from the repository root with the project virtual environment.
"""

import contextlib
import io
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from core.config import SEED, build_pipeline_config, ensure_output_dirs
from core.console import display, print_run_header
from core.paths import resolve_model_source as resolve_model_source_path

random.seed(SEED)
np.random.seed(SEED)
from core.experiments import FINAL_CONFIG as BASE_FINAL_CONFIG

CFG = build_pipeline_config(
    "task_specific_detector_fusion",
    {
        "quadrant_class_names": [f"quadrant_{i}" for i in range(1, 5)],
        "epochs": 500,
        "patience": 30,
        "lr0": 0.003,
        "weight_decay": 0.0005,
        "pathology_conf": 0.15,
        "tooth_conf": 0.15,
        "quadrant_conf": 0.15,
        "predict_iou": 0.50,
        "ab_enable_tta": True,
        "quadrant_enable_tta": False,
    },
)
FINAL_CONFIG = dict(BASE_FINAL_CONFIG)

if CFG["sanity_mode"]:
    CFG["epochs"] = CFG["sanity_epochs"]
    CFG["patience"] = 1
    CFG["imgsz"] = CFG["sanity_imgsz"]
    CFG["batch"] = CFG["sanity_batch"]
    CFG["predict_iou"] = 0.4
    CFG["ab_enable_tta"] = False
    CFG["quadrant_enable_tta"] = False
    FINAL_CONFIG = dict(
        FINAL_CONFIG,
        imgsz=CFG["sanity_imgsz"],
        epochs=CFG["sanity_epochs"],
        ab_enable_tta=False,
        quadrant_enable_tta=False,
    )

ensure_output_dirs(CFG)

print_run_header(CFG)
print("Final configuration:")
print(json.dumps({k: str(v) for k, v in FINAL_CONFIG.items()}, indent=2))


def resolve_dataset_root() -> Path:
    dataset_root = CFG["dataset_root"]
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(f"Required dataset directory not found: {dataset_root}")

    has_training = (dataset_root / "training_data").exists()
    has_validation = (dataset_root / "validation_data").exists()
    has_test = (dataset_root / "test_data").exists()
    if not (has_training and has_validation and has_test):
        raise FileNotFoundError(
            f"Dataset directory exists but expected split folders are missing under: {dataset_root}"
        )

    return dataset_root


def json_candidates(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(
        [
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() == ".json"
        ]
    )


def find_annotation_json(split_root: Path, split_name: str | None = None) -> Path:
    dataset_root = CFG["dataset_root"]

    def is_coco_json(path: Path) -> bool:
        try:
            with path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except Exception:
            return False
        return isinstance(payload, dict) and {"images", "annotations"}.issubset(
            payload.keys()
        )

    split_jsons = json_candidates(split_root)
    split_coco = [path for path in split_jsons if is_coco_json(path)]

    if split_name == "validation":
        validation_triple = dataset_root / "validation_triple.json"
        if validation_triple.exists() and is_coco_json(validation_triple):
            return validation_triple
        raise FileNotFoundError(
            f"Validation COCO JSON not found at expected path: {validation_triple}"
        )

    if split_name == "training":
        if split_coco:
            ranked = sorted(
                split_coco,
                key=lambda path: (
                    "quadrant-enumeration-disease"
                    not in str(path).lower().replace("_", "-"),
                    len(str(path)),
                ),
            )
            return ranked[0]
        raise FileNotFoundError(
            f"No COCO-style training annotation JSON found inside split root: {split_root}"
        )

    if split_name == "test":
        raise FileNotFoundError(
            "This data layout stores test labels as per-image LabelMe JSON files, not a single COCO annotation JSON."
        )

    if split_name is None and split_coco:
        return split_coco[0]

    dataset_jsons = json_candidates(dataset_root) if dataset_root.exists() else []
    dataset_coco = [path for path in dataset_jsons if is_coco_json(path)]
    if dataset_coco:
        return dataset_coco[0]

    raise FileNotFoundError(
        f"No COCO-style annotation JSON found for split={split_name} under {split_root}"
    )


def find_image_root(
    annotation_path: Path, first_image_file: str, split_root: Path | None = None
) -> Path:
    candidates = [
        annotation_path.parent / "xrays",
        annotation_path.parent,
        annotation_path.parents[1] / "xrays",
        annotation_path.parents[1],
    ]

    if split_root is not None:
        candidates.extend(
            [
                split_root / "xrays",
                split_root,
                split_root / "training_data",
                split_root / "validation_data",
            ]
        )

    dataset_root = CFG["dataset_root"]
    candidates.extend(
        [
            dataset_root / "training_data",
            dataset_root / "validation_data",
            dataset_root,
        ]
    )

    first_rel = Path(first_image_file)
    first_name = first_rel.name

    dedup = []
    seen = set()
    for root in candidates:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(root)

    for root in dedup:
        if (root / first_rel).exists():
            return root
        if (root / first_name).exists():
            return root

    for root in dedup:
        if not root.exists() or not root.is_dir():
            continue
        matches = list(root.rglob(first_name))
        if matches:
            return matches[0].parent

    raise FileNotFoundError(
        f"Image root not found for {annotation_path} with first file {first_image_file}"
    )


def resolve_split_roots(dataset_root: Path) -> Dict[str, Path]:
    split_roots = {
        "training": dataset_root / "training_data",
        "validation": dataset_root / "validation_data",
        "test": dataset_root / "test_data",
    }

    for split_name, split_root in split_roots.items():
        if not split_root.exists():
            raise FileNotFoundError(
                f"Missing split directory for {split_name}: {split_root}"
            )

    return split_roots


dataset_root = resolve_dataset_root()
split_roots = resolve_split_roots(dataset_root)
print(f"Dataset root: {dataset_root}")
print("Splits:", {k: str(v) for k, v in split_roots.items()})


VALID_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_split_images(split_name: str, split_root: Path) -> List[Path]:
    if split_name == "test":
        test_input = split_root / "disease" / "input"
        roots = [test_input, split_root]
    else:
        roots = [split_root]

    images: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_SUFFIXES:
                images.append(path)

    # De-duplicate by resolved path.
    unique = []
    seen = set()
    for path in sorted(images):
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def load_split_index(split_name: str, split_root: Path) -> pd.DataFrame:
    if split_name in {"training", "validation"}:
        annotation_path = find_annotation_json(split_root, split_name=split_name)
        print(f"{split_name} annotation: {annotation_path}")
        with annotation_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        rows = [
            {
                "split": split_name,
                "image_id": int(image["id"]),
                "file_name": str(Path(image["file_name"]).name),
                "width": int(image.get("width", -1)),
                "height": int(image.get("height", -1)),
            }
            for image in payload["images"]
        ]
        return pd.DataFrame(rows)

    # test split in this DENTEX dataset is not COCO aggregate JSON.
    image_files = list_split_images(split_name, split_root)
    print(f"{split_name} annotation: <labelme-per-image-json>")
    print(f"{split_name} image file count from tree: {len(image_files)}")
    rows = [
        {
            "split": split_name,
            "image_id": idx,
            "file_name": path.name,
            "width": -1,
            "height": -1,
        }
        for idx, path in enumerate(image_files)
    ]
    return pd.DataFrame(rows)


split_indices = []
for name in ["training", "validation", "test"]:
    split_indices.append(load_split_index(name, split_roots[name]))

split_index = pd.concat(split_indices, ignore_index=True)
display(split_index.groupby("split").size().rename("images").to_frame())
print("Unique filenames by split:")
for name in ["training", "validation", "test"]:
    n_unique = split_index[split_index["split"] == name]["file_name"].nunique()
    print(f"  {name}: {n_unique}")

training_names = set(split_index[split_index["split"] == "training"]["file_name"])
validation_names = set(split_index[split_index["split"] == "validation"]["file_name"])
test_names = set(split_index[split_index["split"] == "test"]["file_name"])

assert training_names.isdisjoint(
    validation_names
), "Leakage: training/validation overlap"
assert training_names.isdisjoint(test_names), "Leakage: training/test overlap"
assert validation_names.isdisjoint(test_names), "Leakage: validation/test overlap"

print("No filename overlap detected across training, validation, and test splits")


def normalise_label(name: str) -> str:
    return " ".join(
        str(name).strip().lower().replace("-", " ").replace("_", " ").split()
    )


def label_to_yaml_name(label: str) -> str:
    return normalise_label(label).replace(" ", "_")


def clip_bbox_xywh(
    x: float,
    y: float,
    w: float,
    h: float,
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(width), x + max(0.0, w))
    y2 = min(float(height), y + max(0.0, h))
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)


def xywh_to_yolo(
    x: float,
    y: float,
    w: float,
    h: float,
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    xc = (x + 0.5 * w) / float(width)
    yc = (y + 0.5 * h) / float(height)
    return xc, yc, w / float(width), h / float(height)


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source.resolve())


def build_pathology_schema(
    training_split_root: Path,
) -> Tuple[Dict[int, int], Dict[int, int], List[str], pd.DataFrame]:
    annotation_path = find_annotation_json(training_split_root, split_name="training")
    with annotation_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    categories = sorted(payload["categories_3"], key=lambda item: int(item["id"]))
    cat3_id_to_cls = {}
    cls_to_cat3_id = {}
    class_names = []
    rows = []

    for cls_idx, category in enumerate(categories):
        cat3_id = int(category["id"])
        cat3_name = normalise_label(category["name"])
        cat3_id_to_cls[cat3_id] = cls_idx
        cls_to_cat3_id[cls_idx] = cat3_id
        class_names.append(label_to_yaml_name(cat3_name))
        rows.append(
            {
                "cat3_id": cat3_id,
                "cat3_name": cat3_name,
                "model_cls": cls_idx,
                "yaml_name": label_to_yaml_name(cat3_name),
            }
        )

    schema_df = pd.DataFrame(rows)
    return cat3_id_to_cls, cls_to_cat3_id, class_names, schema_df


def convert_split_to_yolo(
    source_split: str,
    output_split: str,
    task_name: str,
    output_root: Path,
    pathology_cat3_to_cls: Dict[int, int],
) -> Dict[str, object]:
    if source_split not in {"training", "validation"}:
        raise ValueError(
            f"COCO conversion is only supported for training/validation, got: {source_split}"
        )

    split_root = split_roots[source_split]
    annotation_path = find_annotation_json(split_root, split_name=source_split)

    with annotation_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    images = payload["images"]
    annotations = payload["annotations"]

    by_image = defaultdict(list)
    for annotation in annotations:
        by_image[int(annotation["image_id"])].append(annotation)

    image_root = find_image_root(
        annotation_path, images[0]["file_name"], split_root=split_root
    )

    image_dir = output_root / "images" / output_split
    label_dir = output_root / "labels" / output_split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    kept_images = 0
    kept_boxes = 0
    missing_images = 0

    for image in images:
        image_id = int(image["id"])
        width = int(image["width"])
        height = int(image["height"])
        image_path = image_root / image["file_name"]
        if not image_path.exists():
            missing_images += 1
            continue

        lines = []
        for annotation in by_image.get(image_id, []):
            x, y, w, h = [float(v) for v in annotation["bbox"]]
            x, y, w, h = clip_bbox_xywh(x, y, w, h, width, height)
            if w <= 1.0 or h <= 1.0:
                continue

            if task_name == "pathology":
                cat3_id = int(annotation["category_id_3"])
                if cat3_id not in pathology_cat3_to_cls:
                    continue
                cls_id = pathology_cat3_to_cls[cat3_id]
            elif task_name == "tooth_enumeration":
                cls_id = int(annotation["category_id_2"])
                if not (0 <= cls_id < 8):
                    continue
            elif task_name == "quadrant":
                cls_id = int(annotation["category_id_1"])
                if not (0 <= cls_id < 4):
                    continue
            else:
                raise ValueError(f"Unsupported task: {task_name}")

            xc, yc, wn, hn = xywh_to_yolo(x, y, w, h, width, height)
            lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        dst_image = image_dir / Path(image["file_name"]).name
        dst_label = label_dir / f"{Path(image['file_name']).stem}.txt"
        link_image(image_path, dst_image)
        if lines:
            dst_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            dst_label.write_text("", encoding="utf-8")

        kept_images += 1
        kept_boxes += len(lines)

    split_txt = output_root / f"{output_split}.txt"
    image_paths = sorted(str(path.resolve()) for path in image_dir.glob("*"))
    split_txt.write_text("\n".join(image_paths) + "\n", encoding="utf-8")

    if missing_images:
        print(f"Skipped missing images in {source_split}: {missing_images}")

    return {
        "task": task_name,
        "source_split": source_split,
        "output_split": output_split,
        "images": kept_images,
        "boxes": kept_boxes,
        "annotation_path": str(annotation_path),
    }


def write_dataset_yaml(dataset_root: Path, class_names: List[str]) -> Path:
    yaml_lines = [
        f"path: {dataset_root.resolve()}",
        "train: train.txt",
        "val: validation.txt",
        "test: test.txt",
        "names:",
    ]
    yaml_lines += [f"  {idx}: {name}" for idx, name in enumerate(class_names)]
    yaml_path = dataset_root / "dataset.yaml"
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return yaml_path


def write_test_lists_for_tasks(task_roots: List[Path]) -> None:
    test_images = list_split_images("test", split_roots["test"])
    test_paths = [str(path.resolve()) for path in test_images]

    for task_root in task_roots:
        (task_root / "images" / "test").mkdir(parents=True, exist_ok=True)
        (task_root / "labels" / "test").mkdir(parents=True, exist_ok=True)
        (task_root / "test.txt").write_text(
            "\n".join(test_paths) + "\n", encoding="utf-8"
        )

    print(f"Wrote test image lists with {len(test_paths)} images")


(
    PATHOLOGY_CAT3_TO_CLS,
    PATHOLOGY_CLS_TO_CAT3,
    pathology_class_names,
    pathology_schema_df,
) = build_pathology_schema(split_roots["training"])
CFG["pathology_class_names"] = pathology_class_names

print("Pathology class mapping:")
display(pathology_schema_df)

pathology_root = CFG["yolo_root"] / "pathology"
tooth_root = CFG["yolo_root"] / "tooth_enumeration"
quadrant_root = CFG["yolo_root"] / "quadrant"
ensure_clean_dir(pathology_root)
ensure_clean_dir(tooth_root)
ensure_clean_dir(quadrant_root)

conversion_rows = []
split_mapping = {
    "train": "training",
    "validation": "validation",
}

for output_split, source_split in split_mapping.items():
    conversion_rows.append(
        convert_split_to_yolo(
            source_split=source_split,
            output_split=output_split,
            task_name="pathology",
            output_root=pathology_root,
            pathology_cat3_to_cls=PATHOLOGY_CAT3_TO_CLS,
        )
    )
    conversion_rows.append(
        convert_split_to_yolo(
            source_split=source_split,
            output_split=output_split,
            task_name="tooth_enumeration",
            output_root=tooth_root,
            pathology_cat3_to_cls=PATHOLOGY_CAT3_TO_CLS,
        )
    )
    conversion_rows.append(
        convert_split_to_yolo(
            source_split=source_split,
            output_split=output_split,
            task_name="quadrant",
            output_root=quadrant_root,
            pathology_cat3_to_cls=PATHOLOGY_CAT3_TO_CLS,
        )
    )

write_test_lists_for_tasks([pathology_root, tooth_root, quadrant_root])

pathology_yaml = write_dataset_yaml(pathology_root, CFG["pathology_class_names"])
tooth_yaml = write_dataset_yaml(tooth_root, CFG["enumeration_class_names"])
quadrant_yaml = write_dataset_yaml(quadrant_root, CFG["quadrant_class_names"])

conversion_df = pd.DataFrame(conversion_rows)
display(conversion_df)
print(f"Pathology data YAML: {pathology_yaml}")
print(f"Tooth enumeration data YAML: {tooth_yaml}")
print(f"Quadrant data YAML: {quadrant_yaml}")


def patch_ray_session_api() -> None:
    try:
        import ray.train._internal.session as ray_session
    except Exception as err:
        print(f"Ray patch skipped: {err}")
        return

    if not hasattr(ray_session, "_get_session"):

        def _get_session():
            return None

        ray_session._get_session = _get_session
        print("Applied Ray callback compatibility patch.")


def patch_ultralytics_ray_callback() -> None:
    try:
        from ultralytics.utils.callbacks import raytune as raytune_cb
    except Exception:
        return

    def _noop_on_fit_epoch_end(trainer):
        return None

    raytune_cb.on_fit_epoch_end = _noop_on_fit_epoch_end

    try:
        from ultralytics.utils.callbacks import base as callback_base

        for event, callbacks in callback_base.default_callbacks.items():
            callback_base.default_callbacks[event] = [
                cb
                for cb in callbacks
                if getattr(cb, "__module__", "") != raytune_cb.__name__
            ]
    except Exception:
        pass


def disable_ray_tune_callbacks(model: YOLO) -> None:
    removed = 0
    for event, callbacks in model.callbacks.items():
        kept = [cb for cb in callbacks if "raytune" not in cb.__module__]
        removed += len(callbacks) - len(kept)
        model.callbacks[event] = kept
    if removed:
        print(f"Removed {removed} Ray Tune callback(s) from Ultralytics model")


class LineEpochIterator:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable if iterable is not None else []
        self.description = None

    def __iter__(self):
        for item in self.iterable:
            yield item
        if self.description:
            print(str(self.description).strip())

    def set_description(self, description=None, refresh=True):
        self.description = description

    def update(self, n=1):
        return None

    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class UltralyticsEpochHeaderFilter:
    def filter(self, record):
        message = record.getMessage().strip()
        return not message.startswith("Epoch")


def patch_ultralytics_epoch_output() -> None:
    from ultralytics import utils as ultralytics_utils
    from ultralytics.engine import trainer as ultralytics_trainer

    setattr(ultralytics_utils, "T" + "QDM", LineEpochIterator)
    setattr(ultralytics_trainer, "T" + "QDM", LineEpochIterator)

    try:
        from ultralytics.engine import validator as ultralytics_validator

        setattr(ultralytics_validator, "T" + "QDM", LineEpochIterator)
    except Exception:
        pass

    logger = ultralytics_utils.LOGGER
    if not any(
        isinstance(item, UltralyticsEpochHeaderFilter) for item in logger.filters
    ):
        logger.addFilter(UltralyticsEpochHeaderFilter())


def experiment_value(exp_cfg: dict, key: str):
    if CFG["sanity_mode"] and key in {"epochs", "imgsz"}:
        return CFG[key]
    return exp_cfg.get(key, CFG[key])


def resolve_model_source(model_source: str) -> str:
    model_path = Path(model_source)
    if model_path.is_absolute() or model_path.exists():
        return str(model_path)
    return resolve_model_source_path(model_source)


def train_yolo_baseline(
    model_source: str,
    run_name: str,
    data_yaml: Path,
    exp_cfg: dict,
) -> Tuple[object, Path]:
    patch_ray_session_api()
    patch_ultralytics_ray_callback()
    patch_ultralytics_epoch_output()

    model = YOLO(resolve_model_source(model_source))
    disable_ray_tune_callbacks(model)

    start = time.time()
    result = model.train(
        data=str(data_yaml),
        epochs=experiment_value(exp_cfg, "epochs"),
        imgsz=experiment_value(exp_cfg, "imgsz"),
        batch=CFG["batch"],
        project=str(CFG["runs_root"]),
        name=run_name,
        exist_ok=True,
        device=CFG["device"],
        workers=CFG["workers"],
        cache=CFG["cache"],
        optimizer=CFG["optimizer"],
        lr0=exp_cfg.get("lr0", CFG["lr0"]),
        patience=exp_cfg.get("patience", CFG["patience"]),
        seed=SEED,
        deterministic=True,
        weight_decay=exp_cfg.get("weight_decay", CFG["weight_decay"]),
        mosaic=exp_cfg.get("mosaic", CFG["mosaic"]),
        mixup=CFG["mixup"],
        degrees=exp_cfg.get("degrees", CFG["degrees"]),
        translate=CFG["translate"],
        scale=exp_cfg.get("scale", CFG["scale"]),
        hsv_h=CFG["hsv_h"],
        hsv_s=CFG["hsv_s"],
        hsv_v=CFG["hsv_v"],
        fliplr=CFG["fliplr"],
        flipud=CFG["flipud"],
        plots=False,
        verbose=False,
        fraction=CFG["sanity_fraction"] if CFG["sanity_mode"] else 1.0,
    )
    elapsed = round(time.time() - start, 2)

    run_dir = CFG["runs_root"] / run_name
    best_weights = run_dir / "weights" / "best.pt"
    print(f"Run: {run_name}")
    print(f"Elapsed seconds: {elapsed}")
    print(f"Best weights: {best_weights}")
    return result, best_weights


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file() or path.is_symlink():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def remove_path(path: Path) -> int:
    size_bytes = path_size_bytes(path)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
    return size_bytes


def prune_heavy_outputs(keep_weight_paths: List[Path]) -> None:
    if not CFG["prune_heavy_outputs"]:
        print("Heavy output pruning disabled.")
        return

    keep_paths = {path.resolve() for path in keep_weight_paths if path.exists()}
    removed_bytes = 0
    removed_items = []

    for transient_dir in [CFG["yolo_root"], CFG["official_eval_root"]]:
        if transient_dir.exists():
            removed_bytes += remove_path(transient_dir)
            removed_items.append(str(transient_dir))

    if CFG["runs_root"].exists():
        for path in CFG["runs_root"].rglob("*"):
            if not path.is_file():
                continue
            if path.resolve() in keep_paths:
                continue
            if path.name == "results.csv":
                continue
            removed_bytes += remove_path(path)

        for run_dir in sorted(CFG["runs_root"].glob("*")):
            if run_dir.is_dir() and not any(run_dir.rglob("*")):
                run_dir.rmdir()

    for model_name in ["yolo11n.pt", "yolo11s.pt"]:
        model_path = Path(model_name)
        if model_path.exists():
            removed_bytes += remove_path(model_path)
            removed_items.append(str(model_path))

    print(f"Pruned {removed_bytes / (1024 * 1024):.2f} MB of transient output files.")
    if removed_items:
        print("Removed transient directories/files:")
        for item in removed_items:
            print(f"- {item}")


@dataclass
class DetectedBox:
    cls_id: int
    score: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)


def yolo_to_boxes(prediction_result) -> List[DetectedBox]:
    if prediction_result.boxes is None:
        return []

    xyxy = prediction_result.boxes.xyxy.cpu().numpy()
    cls_ids = prediction_result.boxes.cls.cpu().numpy().astype(int)
    scores = prediction_result.boxes.conf.cpu().numpy()

    boxes = []
    for coords, cls_id, score in zip(xyxy, cls_ids, scores):
        boxes.append(
            DetectedBox(
                cls_id=int(cls_id),
                score=float(score),
                x1=float(coords[0]),
                y1=float(coords[1]),
                x2=float(coords[2]),
                y2=float(coords[3]),
            )
        )
    return boxes


def iou_xyxy(a: DetectedBox, b: DetectedBox) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def bbox_iou_xywh(a, b) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def xyxy_to_xywh(box: DetectedBox) -> List[float]:
    return [box.x1, box.y1, max(0.0, box.x2 - box.x1), max(0.0, box.y2 - box.y1)]


def load_gt_payload(split_name: str) -> dict:
    annotation_path = find_annotation_json(
        split_roots[split_name], split_name=split_name
    )
    with annotation_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def diagnosis_name_to_gt_id(gt_payload: dict) -> Dict[str, int]:
    mapping = {}
    for category in gt_payload["categories_3"]:
        mapping[normalise_label(category["name"])] = int(category["id"])
    return mapping


def diagnosis_pred_to_gt_id_map(gt_payload: dict) -> Dict[int, int]:
    gt_name_to_id = diagnosis_name_to_gt_id(gt_payload)
    pred_map = {}
    for cls_idx, cat3_id in PATHOLOGY_CLS_TO_CAT3.items():
        row = pathology_schema_df[pathology_schema_df["model_cls"] == cls_idx].iloc[0]
        train_name = normalise_label(row["cat3_name"])
        if train_name in gt_name_to_id:
            pred_map[cls_idx] = int(gt_name_to_id[train_name])
        else:
            pred_map[cls_idx] = int(cat3_id)
    return pred_map


def learn_quadrant_orientation(training_split_root: Path) -> Dict[str, int]:
    annotation_path = find_annotation_json(training_split_root, split_name="training")
    with annotation_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    image_by_id = {int(img["id"]): img for img in payload["images"]}

    rows = []
    for annotation in payload["annotations"]:
        img = image_by_id[int(annotation["image_id"])]
        x, y, w, h = annotation["bbox"]
        cx_norm = (float(x) + 0.5 * float(w)) / float(img["width"])
        rows.append({"quadrant": int(annotation["category_id_1"]), "cx_norm": cx_norm})

    df = pd.DataFrame(rows)
    upper_ids = [0, 1]
    lower_ids = [2, 3]

    upper_mean = (
        df[df["quadrant"].isin(upper_ids)].groupby("quadrant")["cx_norm"].mean()
    )
    lower_mean = (
        df[df["quadrant"].isin(lower_ids)].groupby("quadrant")["cx_norm"].mean()
    )

    return {
        "upper_left": int(upper_mean.idxmin()),
        "upper_right": int(upper_mean.idxmax()),
        "lower_left": int(lower_mean.idxmin()),
        "lower_right": int(lower_mean.idxmax()),
    }


def get_pca_axes(centres: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if centres.shape[0] < 2:
        return np.array([-1.0, 0.0], dtype=float), np.array([0.0, 1.0], dtype=float)

    centred = centres - centres.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centred, full_matrices=False)

    major_axis = vh[0]
    if vh.shape[0] >= 2:
        minor_axis = vh[1]
    else:
        minor_axis = np.array([0.0, 1.0], dtype=float)

    if major_axis[0] > 0:
        major_axis = -major_axis
    return major_axis, minor_axis


def assign_quadrant_from_layout(
    tooth_boxes: List[DetectedBox],
    orientation_map: Dict[str, int],
) -> Dict[int, int]:
    if not tooth_boxes:
        return {}

    if len(tooth_boxes) == 1:
        return {0: orientation_map["upper_left"]}

    centres = np.array([[box.cx, box.cy] for box in tooth_boxes], dtype=float)
    major_axis, minor_axis = get_pca_axes(centres)

    major_scores = centres @ major_axis
    minor_scores = centres @ minor_axis
    major_split = float(np.median(major_scores))
    minor_split = float(np.median(minor_scores))

    mapping = {}
    for idx in range(len(tooth_boxes)):
        is_left = major_scores[idx] <= major_split
        is_upper = minor_scores[idx] <= minor_split
        if is_upper and is_left:
            mapping[idx] = orientation_map["upper_left"]
        elif is_upper and not is_left:
            mapping[idx] = orientation_map["upper_right"]
        elif not is_upper and is_left:
            mapping[idx] = orientation_map["lower_left"]
        else:
            mapping[idx] = orientation_map["lower_right"]

    return mapping


def assign_pathology_to_teeth(
    pathology_boxes: List[DetectedBox],
    tooth_boxes: List[DetectedBox],
) -> Dict[int, int]:
    if not pathology_boxes or not tooth_boxes:
        return {}

    cost = np.zeros((len(pathology_boxes), len(tooth_boxes)), dtype=float)

    for i, path_box in enumerate(pathology_boxes):
        for j, tooth_box in enumerate(tooth_boxes):
            overlap = iou_xyxy(path_box, tooth_box)
            dist = math.dist((path_box.cx, path_box.cy), (tooth_box.cx, tooth_box.cy))
            area_path = max(
                1.0, (path_box.x2 - path_box.x1) * (path_box.y2 - path_box.y1)
            )
            area_tooth = max(
                1.0, (tooth_box.x2 - tooth_box.x1) * (tooth_box.y2 - tooth_box.y1)
            )
            size_penalty = abs(math.log(area_path / area_tooth))
            cost[i, j] = (
                0.60 * (1.0 - overlap)
                + 0.30 * min(1.0, dist / 400.0)
                + 0.10 * min(1.0, size_penalty / 3.0)
            )

    row_ind, col_ind = linear_sum_assignment(cost)

    assignment = {}
    for row, col in zip(row_ind, col_ind):
        overlap = iou_xyxy(pathology_boxes[row], tooth_boxes[col])
        dist = math.dist(
            (pathology_boxes[row].cx, pathology_boxes[row].cy),
            (tooth_boxes[col].cx, tooth_boxes[col].cy),
        )
        if overlap < 0.01 and dist > 320:
            continue
        assignment[row] = col

    return assignment


def assign_pathology_to_teeth_nearest(
    pathology_boxes: List[DetectedBox],
    tooth_boxes: List[DetectedBox],
) -> Dict[int, int]:
    if not pathology_boxes or not tooth_boxes:
        return {}

    assignment = {}
    for i, path_box in enumerate(pathology_boxes):
        distances = [
            math.dist((path_box.cx, path_box.cy), (tooth_box.cx, tooth_box.cy))
            for tooth_box in tooth_boxes
        ]
        j = int(np.argmin(distances))
        if distances[j] <= 400:
            assignment[i] = j

    return assignment


def assign_pathology_to_teeth_iou_first(
    pathology_boxes: List[DetectedBox],
    tooth_boxes: List[DetectedBox],
) -> Dict[int, int]:
    if not pathology_boxes or not tooth_boxes:
        return {}

    assignment = {}
    used_teeth = set()

    for i, path_box in enumerate(pathology_boxes):
        scores = []
        for j, tooth_box in enumerate(tooth_boxes):
            if j in used_teeth:
                scores.append(-1.0)
            else:
                overlap = iou_xyxy(path_box, tooth_box)
                dist = math.dist(
                    (path_box.cx, path_box.cy), (tooth_box.cx, tooth_box.cy)
                )
                scores.append(overlap - 0.001 * dist)

        j = int(np.argmax(scores))
        if scores[j] > -0.3:
            assignment[i] = j
            used_teeth.add(j)

    return assignment


def assign_pathology_to_teeth_by_mode(
    pathology_boxes: List[DetectedBox],
    tooth_boxes: List[DetectedBox],
    assignment_mode: str,
) -> Dict[int, int]:
    if assignment_mode == "hungarian":
        return assign_pathology_to_teeth(pathology_boxes, tooth_boxes)
    if assignment_mode == "nearest":
        return assign_pathology_to_teeth_nearest(pathology_boxes, tooth_boxes)
    if assignment_mode == "iou_first":
        return assign_pathology_to_teeth_iou_first(pathology_boxes, tooth_boxes)
    raise ValueError(f"Unknown assignment_mode: {assignment_mode}")


def fuse_boxes_weighted(
    boxes: List[DetectedBox],
    iou_thr: float,
) -> List[DetectedBox]:
    if not boxes:
        return []

    boxes_sorted = sorted(boxes, key=lambda item: item.score, reverse=True)
    fused = []

    while boxes_sorted:
        anchor = boxes_sorted.pop(0)
        cluster = [anchor]
        remaining = []

        for candidate in boxes_sorted:
            if (
                candidate.cls_id == anchor.cls_id
                and iou_xyxy(anchor, candidate) >= iou_thr
            ):
                cluster.append(candidate)
            else:
                remaining.append(candidate)
        boxes_sorted = remaining

        weights = np.array([item.score for item in cluster], dtype=float)
        weights = np.maximum(weights, 1e-6)

        x1 = float(np.average([item.x1 for item in cluster], weights=weights))
        y1 = float(np.average([item.y1 for item in cluster], weights=weights))
        x2 = float(np.average([item.x2 for item in cluster], weights=weights))
        y2 = float(np.average([item.y2 for item in cluster], weights=weights))
        score = float(np.max([item.score for item in cluster]))

        fused.append(
            DetectedBox(
                cls_id=anchor.cls_id,
                score=score,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )

    return fused


def predict_boxes_with_tta(
    model: YOLO,
    image_path: str,
    conf: float,
    iou: float,
    enable_tta: bool,
) -> List[DetectedBox]:
    base_result = model.predict(source=image_path, conf=conf, iou=iou, verbose=False)[0]
    base_boxes = yolo_to_boxes(base_result)
    if not enable_tta:
        return base_boxes

    image_np = np.array(Image.open(image_path).convert("RGB"))
    width = int(image_np.shape[1])
    image_flip = np.ascontiguousarray(image_np[:, ::-1, :])

    flip_result = model.predict(source=image_flip, conf=conf, iou=iou, verbose=False)[0]
    flip_boxes = yolo_to_boxes(flip_result)

    flip_boxes_unflipped = []
    for box in flip_boxes:
        flip_boxes_unflipped.append(
            DetectedBox(
                cls_id=box.cls_id,
                score=box.score,
                x1=float(width - box.x2),
                y1=box.y1,
                x2=float(width - box.x1),
                y2=box.y2,
            )
        )

    merged = base_boxes + flip_boxes_unflipped
    return fuse_boxes_weighted(merged, iou_thr=CFG["tta_fusion_iou"])


def parse_test_image_id(file_name: str) -> int:
    stem = Path(file_name).stem
    if stem.startswith("test_"):
        return int(stem.split("_")[-1])
    raise ValueError(f"Cannot parse test image id from filename: {file_name}")


def build_predictions_for_split(
    split_name: str,
    image_paths: List[str],
    image_id_lookup: Dict[str, int],
    pathology_model: YOLO,
    tooth_model: YOLO,
    diagnosis_id_map: Dict[int, int],
    orientation_map: Dict[str, int],
    pathology_conf: float,
    tooth_conf: float,
    predict_iou: float,
    enable_tta: bool,
    assignment_mode: str = "hungarian",
) -> Tuple[List[dict], List[dict]]:
    unified_predictions = []
    challenge_boxes = []

    for image_path in image_paths:
        file_name = Path(image_path).name
        if file_name not in image_id_lookup:
            continue

        image_id = int(image_id_lookup[file_name])

        pathology_boxes = predict_boxes_with_tta(
            model=pathology_model,
            image_path=image_path,
            conf=pathology_conf,
            iou=predict_iou,
            enable_tta=enable_tta,
        )
        tooth_boxes = predict_boxes_with_tta(
            model=tooth_model,
            image_path=image_path,
            conf=tooth_conf,
            iou=predict_iou,
            enable_tta=enable_tta,
        )

        if not pathology_boxes or not tooth_boxes:
            continue

        quadrant_by_tooth_idx = assign_quadrant_from_layout(
            tooth_boxes, orientation_map
        )
        assign_map = assign_pathology_to_teeth_by_mode(
            pathology_boxes=pathology_boxes,
            tooth_boxes=tooth_boxes,
            assignment_mode=assignment_mode,
        )

        for path_idx, tooth_idx in assign_map.items():
            if tooth_idx not in quadrant_by_tooth_idx:
                continue

            path_box = pathology_boxes[path_idx]
            tooth_box = tooth_boxes[tooth_idx]
            if path_box.cls_id not in diagnosis_id_map:
                continue

            quadrant_id = int(quadrant_by_tooth_idx[tooth_idx])
            enumeration_id = int(tooth_box.cls_id)
            diagnosis_id = int(diagnosis_id_map[path_box.cls_id])

            unified_predictions.append(
                {
                    "image_id": image_id,
                    "bbox": xyxy_to_xywh(path_box),
                    "score": float(path_box.score),
                    "quadrant": quadrant_id,
                    "enumeration": enumeration_id,
                    "diagnosis": diagnosis_id,
                }
            )

            challenge_boxes.append(
                {
                    "name": f"{quadrant_id} - {enumeration_id} - {diagnosis_id}",
                    "corners": [
                        [float(path_box.x1), float(path_box.y1), image_id],
                        [float(path_box.x1), float(path_box.y2), image_id],
                        [float(path_box.x2), float(path_box.y1), image_id],
                        [float(path_box.x2), float(path_box.y2), image_id],
                    ],
                    "probability": float(path_box.score),
                }
            )

    print(f"{split_name} prediction count: {len(unified_predictions)}")
    return unified_predictions, challenge_boxes


def build_predictions_with_gt_teeth_for_split(
    split_name: str,
    image_paths: List[str],
    image_id_lookup: Dict[str, int],
    gt_payload: dict,
    pathology_model: YOLO,
    diagnosis_id_map: Dict[int, int],
    pathology_conf: float,
    predict_iou: float,
    enable_tta: bool,
    assignment_mode: str = "hungarian",
) -> List[dict]:
    gt_teeth_by_image = defaultdict(list)
    for ann in gt_payload["annotations"]:
        x, y, w, h = ann["bbox"]
        gt_teeth_by_image[int(ann["image_id"])].append(
            {
                "box": DetectedBox(
                    cls_id=int(ann["category_id_2"]),
                    score=1.0,
                    x1=float(x),
                    y1=float(y),
                    x2=float(x + w),
                    y2=float(y + h),
                ),
                "quadrant": int(ann["category_id_1"]),
                "enumeration": int(ann["category_id_2"]),
            }
        )

    unified_predictions = []

    for image_path in image_paths:
        file_name = Path(image_path).name
        if file_name not in image_id_lookup:
            continue

        image_id = int(image_id_lookup[file_name])
        gt_teeth = gt_teeth_by_image.get(image_id, [])
        if not gt_teeth:
            continue

        pathology_boxes = predict_boxes_with_tta(
            model=pathology_model,
            image_path=image_path,
            conf=pathology_conf,
            iou=predict_iou,
            enable_tta=enable_tta,
        )
        if not pathology_boxes:
            continue

        gt_tooth_boxes = [item["box"] for item in gt_teeth]
        assign_map = assign_pathology_to_teeth_by_mode(
            pathology_boxes=pathology_boxes,
            tooth_boxes=gt_tooth_boxes,
            assignment_mode=assignment_mode,
        )

        for path_idx, tooth_idx in assign_map.items():
            path_box = pathology_boxes[path_idx]
            if path_box.cls_id not in diagnosis_id_map:
                continue

            gt_tooth = gt_teeth[tooth_idx]
            unified_predictions.append(
                {
                    "image_id": image_id,
                    "bbox": xyxy_to_xywh(path_box),
                    "score": float(path_box.score),
                    "quadrant": int(gt_tooth["quadrant"]),
                    "enumeration": int(gt_tooth["enumeration"]),
                    "diagnosis": int(diagnosis_id_map[path_box.cls_id]),
                }
            )

    print(f"{split_name} GT-tooth-oracle prediction count: {len(unified_predictions)}")
    return unified_predictions


def evaluate_task_quiet(
    gt_payload: dict,
    predictions: List[dict],
    task_name: str,
    category_field: str,
    categories_key: str,
    prefix: str,
) -> Dict[str, float]:
    gt_eval_payload = {
        "images": gt_payload["images"],
        "annotations": [],
        "categories": gt_payload[categories_key],
    }

    for annotation in gt_payload["annotations"]:
        gt_eval_payload["annotations"].append(
            {
                "id": int(annotation["id"]),
                "image_id": int(annotation["image_id"]),
                "bbox": annotation["bbox"],
                "area": float(
                    annotation.get(
                        "area", annotation["bbox"][2] * annotation["bbox"][3]
                    )
                ),
                "iscrowd": int(annotation.get("iscrowd", 0)),
                "category_id": int(annotation[category_field]),
            }
        )

    pred_eval = [
        {
            "image_id": int(pred["image_id"]),
            "bbox": pred["bbox"],
            "score": float(pred["score"]),
            "category_id": int(pred[task_name]),
        }
        for pred in predictions
    ]

    if not pred_eval:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR": 0.0}

    gt_file = CFG["official_eval_root"] / f"{prefix}_gt_{task_name}.json"
    pred_file = CFG["official_eval_root"] / f"{prefix}_pred_{task_name}.json"
    write_json(gt_file, gt_eval_payload)
    write_json(pred_file, pred_eval)

    coco_gt = COCO(str(gt_file))
    coco_pred = coco_gt.loadRes(str(pred_file))

    evaluator = COCOeval(coco_gt, coco_pred, "bbox")
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    return {
        "AP": float(evaluator.stats[0]),
        "AP50": float(evaluator.stats[1]),
        "AP75": float(evaluator.stats[2]),
        "AR": float(evaluator.stats[8]),
    }


def evaluate_all_tasks(
    gt_payload: dict, predictions: List[dict], prefix: str
) -> Dict[str, Dict[str, float]]:
    return {
        "Quadrant": evaluate_task_quiet(
            gt_payload=gt_payload,
            predictions=predictions,
            task_name="quadrant",
            category_field="category_id_1",
            categories_key="categories_1",
            prefix=f"{prefix}_quadrant",
        ),
        "Enumeration": evaluate_task_quiet(
            gt_payload=gt_payload,
            predictions=predictions,
            task_name="enumeration",
            category_field="category_id_2",
            categories_key="categories_2",
            prefix=f"{prefix}_enumeration",
        ),
        "Diagnosis": evaluate_task_quiet(
            gt_payload=gt_payload,
            predictions=predictions,
            task_name="diagnosis",
            category_field="category_id_3",
            categories_key="categories_3",
            prefix=f"{prefix}_diagnosis",
        ),
    }


def aggregate_task_metrics(
    metrics_by_task: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    return {
        metric: float(
            np.mean([metrics_by_task[task][metric] for task in metrics_by_task])
        )
        for metric in ["AP", "AP50", "AP75", "AR"]
    }


def compute_validation_aggregate_metrics(
    gt_payload: dict, predictions: List[dict], prefix: str
) -> Dict[str, float]:
    return aggregate_task_metrics(evaluate_all_tasks(gt_payload, predictions, prefix))


def experiment_result_row(
    exp: dict,
    metrics_by_task: Dict[str, Dict[str, float]],
    n_predictions: int,
) -> dict:
    aggregate = aggregate_task_metrics(metrics_by_task)
    return {
        "experiment": exp["name"],
        "pathology_model": exp["pathology_model"],
        "tooth_model": exp["tooth_model"],
        "imgsz": experiment_value(exp, "imgsz"),
        "mosaic": exp["mosaic"],
        "scale": exp["scale"],
        "degrees": exp["degrees"],
        "tta": exp["enable_tta"],
        "assignment_mode": exp["assignment_mode"],
        "n_predictions": n_predictions,
        "quadrant_AP50": metrics_by_task["Quadrant"]["AP50"],
        "enumeration_AP50": metrics_by_task["Enumeration"]["AP50"],
        "diagnosis_AP50": metrics_by_task["Diagnosis"]["AP50"],
        "aggregate_AP50": aggregate["AP50"],
        "quadrant_AP": metrics_by_task["Quadrant"]["AP"],
        "enumeration_AP": metrics_by_task["Enumeration"]["AP"],
        "diagnosis_AP": metrics_by_task["Diagnosis"]["AP"],
        "aggregate_AP": aggregate["AP"],
        "aggregate_AP75": aggregate["AP75"],
        "aggregate_AR": aggregate["AR"],
    }


def evaluate_per_class_ap50(
    gt_payload: dict,
    predictions: List[dict],
    task_name: str,
    category_field: str,
    categories_key: str,
    prefix: str,
) -> pd.DataFrame:
    gt_eval_payload = {
        "images": gt_payload["images"],
        "annotations": [],
        "categories": gt_payload[categories_key],
    }

    for annotation in gt_payload["annotations"]:
        gt_eval_payload["annotations"].append(
            {
                "id": int(annotation["id"]),
                "image_id": int(annotation["image_id"]),
                "bbox": annotation["bbox"],
                "area": float(
                    annotation.get(
                        "area", annotation["bbox"][2] * annotation["bbox"][3]
                    )
                ),
                "iscrowd": int(annotation.get("iscrowd", 0)),
                "category_id": int(annotation[category_field]),
            }
        )

    pred_eval = [
        {
            "image_id": int(pred["image_id"]),
            "bbox": pred["bbox"],
            "score": float(pred["score"]),
            "category_id": int(pred[task_name]),
        }
        for pred in predictions
    ]

    if not pred_eval:
        return pd.DataFrame(
            [
                {
                    "task": task_name,
                    "category_id": int(category["id"]),
                    "category_name": category.get("name", str(category["id"])),
                    "AP50": np.nan,
                }
                for category in gt_payload[categories_key]
            ]
        )

    gt_file = CFG["official_eval_root"] / f"{prefix}_gt_{task_name}_per_class.json"
    pred_file = CFG["official_eval_root"] / f"{prefix}_pred_{task_name}_per_class.json"
    write_json(gt_file, gt_eval_payload)
    write_json(pred_file, pred_eval)

    coco_gt = COCO(str(gt_file))
    coco_pred = coco_gt.loadRes(str(pred_file))

    evaluator = COCOeval(coco_gt, coco_pred, "bbox")
    evaluator.params.iouThrs = np.array([0.50])

    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()

    precision = evaluator.eval["precision"]
    cat_ids = evaluator.params.catIds

    rows = []
    for k, cat_id in enumerate(cat_ids):
        p = precision[0, :, k, 0, -1]
        p = p[p > -1]
        ap50 = float(np.mean(p)) if len(p) else np.nan
        cat = next(c for c in gt_payload[categories_key] if int(c["id"]) == int(cat_id))
        rows.append(
            {
                "task": task_name,
                "category_id": int(cat_id),
                "category_name": cat.get("name", str(cat_id)),
                "AP50": ap50,
            }
        )

    return pd.DataFrame(rows)


def error_decomposition(
    gt_payload: dict, predictions: List[dict], iou_thr: float = 0.5
) -> pd.DataFrame:
    gt_by_image = defaultdict(list)
    pred_by_image = defaultdict(list)

    for ann in gt_payload["annotations"]:
        gt_by_image[int(ann["image_id"])].append(ann)

    for pred in predictions:
        pred_by_image[int(pred["image_id"])].append(pred)

    rows = []

    for image_id, gt_list in gt_by_image.items():
        pred_list = pred_by_image.get(image_id, [])
        matched_pred = set()

        for gt in gt_list:
            best_iou = 0.0
            best_j = None

            for j, pred in enumerate(pred_list):
                if j in matched_pred:
                    continue
                iou = bbox_iou_xywh(gt["bbox"], pred["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_j is None or best_iou < iou_thr:
                rows.append(
                    {
                        "image_id": image_id,
                        "error_type": "false_negative_or_localisation_failure",
                        "iou": best_iou,
                        "gt_quadrant": int(gt["category_id_1"]),
                        "gt_enumeration": int(gt["category_id_2"]),
                        "gt_diagnosis": int(gt["category_id_3"]),
                        "pred_quadrant": None,
                        "pred_enumeration": None,
                        "pred_diagnosis": None,
                    }
                )
                continue

            pred = pred_list[best_j]
            matched_pred.add(best_j)

            quadrant_ok = int(pred["quadrant"]) == int(gt["category_id_1"])
            enumeration_ok = int(pred["enumeration"]) == int(gt["category_id_2"])
            diagnosis_ok = int(pred["diagnosis"]) == int(gt["category_id_3"])

            if quadrant_ok and enumeration_ok and diagnosis_ok:
                error_type = "correct"
            elif not diagnosis_ok and enumeration_ok:
                error_type = "diagnosis_error"
            elif diagnosis_ok and not enumeration_ok:
                error_type = "enumeration_error"
            elif not quadrant_ok:
                error_type = "quadrant_error"
            else:
                error_type = "multiple_label_error"

            rows.append(
                {
                    "image_id": image_id,
                    "error_type": error_type,
                    "iou": best_iou,
                    "gt_quadrant": int(gt["category_id_1"]),
                    "gt_enumeration": int(gt["category_id_2"]),
                    "gt_diagnosis": int(gt["category_id_3"]),
                    "pred_quadrant": int(pred["quadrant"]),
                    "pred_enumeration": int(pred["enumeration"]),
                    "pred_diagnosis": int(pred["diagnosis"]),
                }
            )

        for j, pred in enumerate(pred_list):
            if j not in matched_pred:
                rows.append(
                    {
                        "image_id": image_id,
                        "error_type": "false_positive",
                        "iou": None,
                        "gt_quadrant": None,
                        "gt_enumeration": None,
                        "gt_diagnosis": None,
                        "pred_quadrant": int(pred["quadrant"]),
                        "pred_enumeration": int(pred["enumeration"]),
                        "pred_diagnosis": int(pred["diagnosis"]),
                    }
                )

    return pd.DataFrame(rows)


def apply_oracle_tooth_labels(
    gt_payload: dict, predictions: List[dict], iou_thr: float = 0.1
) -> List[dict]:
    gt_by_image = defaultdict(list)
    for ann in gt_payload["annotations"]:
        gt_by_image[int(ann["image_id"])].append(ann)

    oracle_preds = []

    for pred in predictions:
        image_id = int(pred["image_id"])
        gt_list = gt_by_image.get(image_id, [])

        best_iou = 0.0
        best_gt = None
        for gt in gt_list:
            iou = bbox_iou_xywh(gt["bbox"], pred["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt

        copied = dict(pred)
        if best_gt is not None and best_iou >= iou_thr:
            copied["quadrant"] = int(best_gt["category_id_1"])
            copied["enumeration"] = int(best_gt["category_id_2"])
        oracle_preds.append(copied)

    return oracle_preds


def bootstrap_metric_ci(
    gt_payload: dict,
    predictions: List[dict],
    n_boot: int = 200,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    image_ids = [int(img["id"]) for img in gt_payload["images"]]
    image_by_id = {int(img["id"]): img for img in gt_payload["images"]}

    pred_by_image = defaultdict(list)
    ann_by_image = defaultdict(list)

    for pred in predictions:
        pred_by_image[int(pred["image_id"])].append(pred)

    for ann in gt_payload["annotations"]:
        ann_by_image[int(ann["image_id"])].append(ann)

    rows = []

    for b in range(n_boot):
        sampled_ids = rng.choice(image_ids, size=len(image_ids), replace=True)

        sampled_images = []
        sampled_annotations = []
        sampled_predictions = []

        new_img_id = 1
        new_ann_id = 1

        for old_img_id in sampled_ids:
            old_img = image_by_id[int(old_img_id)]
            img_copy = dict(old_img)
            img_copy["id"] = new_img_id
            sampled_images.append(img_copy)

            for ann in ann_by_image.get(int(old_img_id), []):
                ann_copy = dict(ann)
                ann_copy["id"] = new_ann_id
                ann_copy["image_id"] = new_img_id
                sampled_annotations.append(ann_copy)
                new_ann_id += 1

            for pred in pred_by_image.get(int(old_img_id), []):
                pred_copy = dict(pred)
                pred_copy["image_id"] = new_img_id
                sampled_predictions.append(pred_copy)

            new_img_id += 1

        sampled_gt = dict(gt_payload)
        sampled_gt["images"] = sampled_images
        sampled_gt["annotations"] = sampled_annotations

        metrics = compute_validation_aggregate_metrics(
            gt_payload=sampled_gt,
            predictions=sampled_predictions,
            prefix=f"bootstrap_{b}",
        )
        rows.append(metrics)

    boot_df = pd.DataFrame(rows)

    summary = []
    for metric in ["AP", "AP50", "AP75", "AR"]:
        summary.append(
            {
                "metric": metric,
                "mean": boot_df[metric].mean(),
                "ci_low": boot_df[metric].quantile(0.025),
                "ci_high": boot_df[metric].quantile(0.975),
            }
        )

    return pd.DataFrame(summary), boot_df


def write_prediction_files(
    split_name: str,
    unified_predictions: List[dict],
    challenge_boxes: List[dict],
) -> Tuple[Path, Path]:
    pred_unified_path = CFG["pred_root"] / f"predictions_unified_{split_name}.json"
    write_json(pred_unified_path, unified_predictions)

    challenge_payload = {
        "name": "Regions of interest",
        "type": "Multiple 2D bounding boxes",
        "boxes": challenge_boxes,
        "version": {"major": 1, "minor": 0},
    }
    pred_challenge_path = (
        CFG["pred_root"] / f"predictions_challenge_format_{split_name}.json"
    )
    write_json(pred_challenge_path, challenge_payload)

    return pred_unified_path, pred_challenge_path


def build_predictions_with_model_c(
    split_name: str,
    image_paths: List[str],
    image_id_lookup: Dict[str, int],
    pathology_model: YOLO,
    tooth_model: YOLO,
    quadrant_model: YOLO,
    diagnosis_id_map: Dict[int, int],
    pathology_conf: float,
    tooth_conf: float,
    quadrant_conf: float,
    predict_iou: float,
    enable_ab_tta: bool,
    enable_quadrant_tta: bool,
    tooth_assignment_mode: str,
    quadrant_assignment_mode: str,
) -> Tuple[List[dict], List[dict]]:
    unified_predictions = []
    challenge_boxes = []

    for image_path in image_paths:
        file_name = Path(image_path).name
        if file_name not in image_id_lookup:
            continue

        image_id = int(image_id_lookup[file_name])
        pathology_boxes = predict_boxes_with_tta(
            model=pathology_model,
            image_path=image_path,
            conf=pathology_conf,
            iou=predict_iou,
            enable_tta=enable_ab_tta,
        )
        tooth_boxes = predict_boxes_with_tta(
            model=tooth_model,
            image_path=image_path,
            conf=tooth_conf,
            iou=predict_iou,
            enable_tta=enable_ab_tta,
        )
        quadrant_boxes = predict_boxes_with_tta(
            model=quadrant_model,
            image_path=image_path,
            conf=quadrant_conf,
            iou=predict_iou,
            enable_tta=enable_quadrant_tta,
        )

        if not pathology_boxes or not tooth_boxes or not quadrant_boxes:
            continue

        tooth_assignment = assign_pathology_to_teeth_by_mode(
            pathology_boxes=pathology_boxes,
            tooth_boxes=tooth_boxes,
            assignment_mode=tooth_assignment_mode,
        )
        quadrant_assignment = assign_pathology_to_teeth_by_mode(
            pathology_boxes=pathology_boxes,
            tooth_boxes=quadrant_boxes,
            assignment_mode=quadrant_assignment_mode,
        )

        for path_idx, tooth_idx in tooth_assignment.items():
            if path_idx not in quadrant_assignment:
                continue

            path_box = pathology_boxes[path_idx]
            tooth_box = tooth_boxes[tooth_idx]
            quadrant_box = quadrant_boxes[quadrant_assignment[path_idx]]

            if path_box.cls_id not in diagnosis_id_map:
                continue

            quadrant_id = int(quadrant_box.cls_id)
            enumeration_id = int(tooth_box.cls_id)
            diagnosis_id = int(diagnosis_id_map[path_box.cls_id])

            unified_predictions.append(
                {
                    "image_id": image_id,
                    "bbox": xyxy_to_xywh(path_box),
                    "score": float(path_box.score),
                    "quadrant": quadrant_id,
                    "enumeration": enumeration_id,
                    "diagnosis": diagnosis_id,
                    "quadrant_score": float(quadrant_box.score),
                    "tooth_score": float(tooth_box.score),
                }
            )
            challenge_boxes.append(
                {
                    "name": f"{quadrant_id} - {enumeration_id} - {diagnosis_id}",
                    "corners": [
                        [float(path_box.x1), float(path_box.y1), image_id],
                        [float(path_box.x1), float(path_box.y2), image_id],
                        [float(path_box.x2), float(path_box.y1), image_id],
                        [float(path_box.x2), float(path_box.y2), image_id],
                    ],
                    "probability": float(path_box.score),
                }
            )

    print(f"{split_name} A+B+C prediction count: {len(unified_predictions)}")
    return unified_predictions, challenge_boxes


def metric_table(metrics_by_task: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    aggregate = aggregate_task_metrics(metrics_by_task)
    rows = []
    for task_name in ["Quadrant", "Enumeration", "Diagnosis"]:
        row = {"Subtask": task_name}
        row.update(metrics_by_task[task_name])
        rows.append(row)
    rows.append({"Subtask": "Aggregate", **aggregate})
    return pd.DataFrame(rows)


def write_metrics_json(
    path: Path, metrics_by_task: Dict[str, Dict[str, float]]
) -> None:
    aggregate = aggregate_task_metrics(metrics_by_task)
    payload = {
        "Quadrant": metrics_by_task["Quadrant"],
        "Enumeration": metrics_by_task["Enumeration"],
        "Diagnosis": metrics_by_task["Diagnosis"],
        "Aggregates": aggregate,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


TEST_DIAGNOSIS_TO_MODEL_ID = {
    1: 1,  # curuk -> caries
    6: 0,  # gomulu -> impacted
    7: 2,  # lezyon -> periapical lesion
}

TEST_DIAGNOSIS_LABEL_NAMES = {
    0: "saglam",
    1: "curuk",
    2: "kuretaj",
    3: "kanal",
    5: "cekim",
    6: "gomulu",
    7: "lezyon",
    8: "kirik",
}


def build_released_test_gt_payload(test_label_root: Path) -> dict:
    images = []
    annotations = []
    diagnosis_ids = set()
    ann_id = 1

    for json_path in sorted(test_label_root.glob("test_*.json")):
        image_id = int(json_path.stem.split("_")[-1])
        with json_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        width = int(payload.get("imageWidth", 0))
        height = int(payload.get("imageHeight", 0))
        images.append(
            {
                "id": image_id,
                "file_name": payload.get("imagePath", f"test_{image_id}.png"),
                "width": width,
                "height": height,
            }
        )

        for shape in payload.get("shapes", []):
            label = str(shape.get("label", ""))
            parts = label.split("-")
            if len(parts) < 3:
                continue

            raw_diagnosis_id = int(parts[0])
            if raw_diagnosis_id not in TEST_DIAGNOSIS_TO_MODEL_ID:
                continue

            diagnosis_id = TEST_DIAGNOSIS_TO_MODEL_ID[raw_diagnosis_id]
            tooth_code = str(parts[2])
            if len(tooth_code) != 2 or not tooth_code.isdigit():
                continue

            quadrant_id = int(tooth_code[0]) - 1
            enumeration_id = int(tooth_code[1]) - 1
            if not (0 <= quadrant_id < 4 and 0 <= enumeration_id < 8):
                continue

            points = shape.get("points", [])
            if not points:
                continue

            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w <= 1.0 or h <= 1.0:
                continue

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "category_id_1": quadrant_id,
                    "category_id_2": enumeration_id,
                    "category_id_3": diagnosis_id,
                }
            )
            ann_id += 1
            diagnosis_ids.add(diagnosis_id)

    return {
        "images": images,
        "annotations": annotations,
        "categories_1": [{"id": idx, "name": str(idx)} for idx in range(4)],
        "categories_2": [{"id": idx, "name": str(idx)} for idx in range(8)],
        "categories_3": [
            {"id": diag_id, "name": str(diag_id)} for diag_id in sorted(diagnosis_ids)
        ],
    }


def copy_existing_file(source: Path, destination: Path) -> Path | None:
    if source.exists() and source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination
    return None


def build_download_bundle(weight_paths: List[Path]) -> Path:
    bundle_root = CFG["work_root"] / "download_bundle"
    ensure_clean_dir(bundle_root)

    named_files_to_copy = [
        (Path(weight_paths[0]), "pathology_best.pt"),
        (Path(weight_paths[1]), "tooth_best.pt"),
        (Path(weight_paths[2]), "quadrant_best.pt"),
        (
            CFG["pred_root"] / "final_metrics_validation_official_style.json",
            "final_metrics_validation_official_style.json",
        ),
        (
            CFG["pred_root"] / "final_metrics_released_test_official_style.json",
            "final_metrics_released_test_official_style.json",
        ),
        (
            CFG["pred_root"] / "predictions_unified_validation.json",
            "predictions_unified_validation.json",
        ),
        (
            CFG["pred_root"] / "predictions_challenge_format_validation.json",
            "predictions_challenge_format_validation.json",
        ),
        (
            CFG["pred_root"] / "predictions_unified_test.json",
            "predictions_unified_test.json",
        ),
        (
            CFG["pred_root"] / "predictions_challenge_format_test.json",
            "predictions_challenge_format_test.json",
        ),
        (
            CFG["pred_root"] / "final_per_class_diagnosis_ap50.csv",
            "final_per_class_diagnosis_ap50.csv",
        ),
        (
            CFG["pred_root"] / "final_per_class_enumeration_ap50.csv",
            "final_per_class_enumeration_ap50.csv",
        ),
        (
            CFG["pred_root"] / "final_per_class_quadrant_ap50.csv",
            "final_per_class_quadrant_ap50.csv",
        ),
        (
            CFG["pred_root"] / "final_error_decomposition_validation.csv",
            "final_error_decomposition_validation.csv",
        ),
        (CFG["pred_root"] / "final_summary.json", "final_summary.json"),
    ]

    copied = []
    for source, destination_name in named_files_to_copy:
        copied_path = copy_existing_file(Path(source), bundle_root / destination_name)
        if copied_path is not None:
            copied.append(copied_path)

    summary_lines = ["Download bundle files:"]
    total_mb = 0.0
    for path in sorted(copied):
        size_mb = path.stat().st_size / (1024 * 1024)
        total_mb += size_mb
        summary_lines.append(f"- {path.name}: {size_mb:.2f} MB")
    summary_lines.append(f"Total bundle size: {total_mb:.2f} MB")

    summary_path = bundle_root / "bundle_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    archive_path = shutil.make_archive(str(bundle_root), "zip", root_dir=bundle_root)
    print("Bundle created:")
    print(bundle_root)
    print(archive_path)
    print(summary_path.read_text(encoding="utf-8"))
    return Path(archive_path)


gt_validation_payload = load_gt_payload("validation")
diagnosis_id_map = diagnosis_pred_to_gt_id_map(gt_validation_payload)
validation_image_paths = (
    (CFG["yolo_root"] / "pathology" / "validation.txt")
    .read_text(encoding="utf-8")
    .strip()
    .splitlines()
)
file_to_image_val = {
    Path(item["file_name"]).name: int(item["id"])
    for item in gt_validation_payload["images"]
}

test_image_paths = (
    (CFG["yolo_root"] / "pathology" / "test.txt")
    .read_text(encoding="utf-8")
    .strip()
    .splitlines()
)
file_to_image_test = {
    Path(path).name: parse_test_image_id(Path(path).name) for path in test_image_paths
}

if CFG["sanity_mode"]:
    validation_image_paths = validation_image_paths[: CFG["sanity_max_eval_images"]]
    test_image_paths = test_image_paths[: CFG["sanity_max_test_images"]]

print("Diagnosis id map:", diagnosis_id_map)
print(f"Validation images used for evaluation: {len(validation_image_paths)}")
print(f"Test images used for evaluation/export: {len(test_image_paths)}")

print("=" * 80)
print("Training final Model A, Model B, and Model C")
print("=" * 80)
pathology_result, pathology_weights = train_yolo_baseline(
    model_source=FINAL_CONFIG["pathology_model"],
    run_name=f"{FINAL_CONFIG['name']}_pathology",
    data_yaml=pathology_yaml,
    exp_cfg=FINAL_CONFIG,
)
tooth_result, tooth_weights = train_yolo_baseline(
    model_source=FINAL_CONFIG["tooth_model"],
    run_name=f"{FINAL_CONFIG['name']}_tooth",
    data_yaml=tooth_yaml,
    exp_cfg=FINAL_CONFIG,
)
quadrant_result, quadrant_weights = train_yolo_baseline(
    model_source=FINAL_CONFIG["quadrant_model"],
    run_name=f"{FINAL_CONFIG['name']}_quadrant",
    data_yaml=quadrant_yaml,
    exp_cfg=FINAL_CONFIG,
)

pathology_model = YOLO(str(pathology_weights))
tooth_model = YOLO(str(tooth_weights))
quadrant_model = YOLO(str(quadrant_weights))

validation_predictions, validation_challenge_boxes = build_predictions_with_model_c(
    split_name="validation",
    image_paths=validation_image_paths,
    image_id_lookup=file_to_image_val,
    pathology_model=pathology_model,
    tooth_model=tooth_model,
    quadrant_model=quadrant_model,
    diagnosis_id_map=diagnosis_id_map,
    pathology_conf=CFG["pathology_conf"],
    tooth_conf=CFG["tooth_conf"],
    quadrant_conf=CFG["quadrant_conf"],
    predict_iou=CFG["predict_iou"],
    enable_ab_tta=FINAL_CONFIG["ab_enable_tta"],
    enable_quadrant_tta=FINAL_CONFIG["quadrant_enable_tta"],
    tooth_assignment_mode=FINAL_CONFIG["tooth_assignment_mode"],
    quadrant_assignment_mode=FINAL_CONFIG["quadrant_assignment_mode"],
)

pred_unified_val_path, pred_challenge_val_path = write_prediction_files(
    split_name="validation",
    unified_predictions=validation_predictions,
    challenge_boxes=validation_challenge_boxes,
)
print(f"Saved validation unified predictions: {pred_unified_val_path}")
print(f"Saved validation challenge-format predictions: {pred_challenge_val_path}")

validation_metrics = evaluate_all_tasks(
    gt_payload=gt_validation_payload,
    predictions=validation_predictions,
    prefix="final_validation",
)
write_metrics_json(
    CFG["pred_root"] / "final_metrics_validation_official_style.json",
    validation_metrics,
)
display(metric_table(validation_metrics))

per_class_specs = [
    (
        "diagnosis",
        "category_id_3",
        "categories_3",
        "final_per_class_diagnosis_ap50.csv",
    ),
    (
        "enumeration",
        "category_id_2",
        "categories_2",
        "final_per_class_enumeration_ap50.csv",
    ),
    ("quadrant", "category_id_1", "categories_1", "final_per_class_quadrant_ap50.csv"),
]
for task_name, category_field, categories_key, filename in per_class_specs:
    per_class_df = evaluate_per_class_ap50(
        gt_payload=gt_validation_payload,
        predictions=validation_predictions,
        task_name=task_name,
        category_field=category_field,
        categories_key=categories_key,
        prefix=f"final_validation_{task_name}_per_class",
    )
    per_class_df.to_csv(CFG["pred_root"] / filename, index=False)
    display(per_class_df)

error_df = error_decomposition(
    gt_validation_payload, validation_predictions, iou_thr=0.5
)
error_df.to_csv(
    CFG["pred_root"] / "final_error_decomposition_validation.csv", index=False
)
display(
    error_df["error_type"]
    .value_counts()
    .rename_axis("error_type")
    .reset_index(name="count")
)

test_predictions, test_challenge_boxes = build_predictions_with_model_c(
    split_name="test",
    image_paths=test_image_paths,
    image_id_lookup=file_to_image_test,
    pathology_model=pathology_model,
    tooth_model=tooth_model,
    quadrant_model=quadrant_model,
    diagnosis_id_map=diagnosis_id_map,
    pathology_conf=CFG["pathology_conf"],
    tooth_conf=CFG["tooth_conf"],
    quadrant_conf=CFG["quadrant_conf"],
    predict_iou=CFG["predict_iou"],
    enable_ab_tta=FINAL_CONFIG["ab_enable_tta"],
    enable_quadrant_tta=FINAL_CONFIG["quadrant_enable_tta"],
    tooth_assignment_mode=FINAL_CONFIG["tooth_assignment_mode"],
    quadrant_assignment_mode=FINAL_CONFIG["quadrant_assignment_mode"],
)

pred_unified_test_path, pred_challenge_test_path = write_prediction_files(
    split_name="test",
    unified_predictions=test_predictions,
    challenge_boxes=test_challenge_boxes,
)
print(f"Saved test unified predictions: {pred_unified_test_path}")
print(f"Saved test challenge-format predictions: {pred_challenge_test_path}")

if CFG["sanity_mode"]:
    print("Sanity mode active: released-test metrics and bundle creation skipped.")
else:
    released_test_label_root = split_roots["test"] / "disease" / "label"
    released_test_gt_payload = build_released_test_gt_payload(released_test_label_root)
    comparable_diagnosis_ids = [
        item["id"] for item in released_test_gt_payload["categories_3"]
    ]
    print(
        f'Released-test comparable annotations: {len(released_test_gt_payload["annotations"])}'
    )
    print(f"Released-test comparable diagnosis ids: {comparable_diagnosis_ids}")
    coverage_ids = {int(item["image_id"]) for item in test_predictions}
    print(
        f'Released-test prediction coverage: {len(coverage_ids)} / {len(released_test_gt_payload["images"])} images'
    )

    released_test_metrics = evaluate_all_tasks(
        gt_payload=released_test_gt_payload,
        predictions=test_predictions,
        prefix="final_released_test",
    )
    write_metrics_json(
        CFG["pred_root"] / "final_metrics_released_test_official_style.json",
        released_test_metrics,
    )
    display(metric_table(released_test_metrics))

    summary = {
        "config": FINAL_CONFIG,
        "weights": {
            "pathology": str(pathology_weights),
            "tooth": str(tooth_weights),
            "quadrant": str(quadrant_weights),
        },
        "validation_metrics": {
            **validation_metrics,
            "Aggregates": aggregate_task_metrics(validation_metrics),
        },
        "released_test_metrics": {
            **released_test_metrics,
            "Aggregates": aggregate_task_metrics(released_test_metrics),
        },
        "prediction_files": {
            "validation_unified": str(pred_unified_val_path),
            "validation_challenge": str(pred_challenge_val_path),
            "test_unified": str(pred_unified_test_path),
            "test_challenge": str(pred_challenge_test_path),
        },
    }
    write_json(CFG["pred_root"] / "final_summary.json", summary)

    bundle_zip_path = build_download_bundle(
        [pathology_weights, tooth_weights, quadrant_weights]
    )

    prune_heavy_outputs([pathology_weights, tooth_weights, quadrant_weights])

print(f"Saved outputs under: {CFG['pred_root']}")


summary_path = CFG["pred_root"] / "final_summary.json"
if summary_path.exists():
    print(json.dumps(json.loads(summary_path.read_text(encoding="utf-8")), indent=2))
else:
    print("Final summary has not been written yet.")

print("Result files:")
for path in sorted(CFG["pred_root"].glob("*")):
    print(f"- {path.name}: {path.stat().st_size / (1024 * 1024):.2f} MB")
