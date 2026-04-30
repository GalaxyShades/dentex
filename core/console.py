"""Console helpers used by local pipeline scripts."""

from __future__ import annotations


def display(obj) -> None:
    if hasattr(obj, "to_string"):
        print(obj.to_string())
    else:
        print(obj)


def print_run_header(cfg: dict, experiment_count: int | None = None) -> None:
    print(f"Local runtime: {cfg['is_local']}")
    print(f"Device: {cfg['device']}")
    print(f"Workers: {cfg['workers']}")
    print(f"Working root: {cfg['work_root']}")
    print(f"Dataset root: {cfg['dataset_root']}")
    print(f"Sanity mode: {cfg['sanity_mode']}")
    if experiment_count is not None:
        print(f"Experiment count: {experiment_count}")
