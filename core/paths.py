"""Repository path helpers for local DENTEX experiments."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "dentex"
MODELS_ROOT = PROJECT_ROOT / "models"
RESULTS_ROOT = PROJECT_ROOT / "results"


def ensure_project_dirs() -> None:
    for path in [DATA_ROOT, MODELS_ROOT, RESULTS_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def resolve_model_source(model_source: str) -> str:
    model_path = Path(model_source)
    if model_path.is_absolute() or model_path.exists():
        return str(model_path)
    return str(MODELS_ROOT / model_source)
