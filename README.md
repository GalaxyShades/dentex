# DENTEX Local Pipelines

This repository has been refactored away from notebook-driven execution into local Python pipelines. Shared configuration, paths, experiment definitions, and command-line utilities now live under `core/`.

## Layout

```text
data/dentex/       Local DENTEX dataset root
models/            YOLO base weights and selected trained weights
notebooks/         Exploratory notebooks and generated figures
pipelines/         Python pipeline scripts converted from the notebooks
results/           Organised prediction, metric, bundle, and training-curve artefacts
core/              Shared research code: paths, configs, experiments, CLI
pyproject.toml     Local package metadata and tooling configuration
requirements.txt   Python runtime requirements
```

The pipeline scripts run locally and do not depend on Kaggle. They expect the dataset under `data/dentex` and weights under `models`.

## Environment

The project virtual environment is `.venv`. It has been checked with:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

To recreate it from scratch:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

The same setup is available through `make init`. Existing environments can be refreshed with `make install`.
## Benchmarking
We conducted a multi-stage benchmarking process to establish a "Pareto Frontier" between diagnostic accuracy and real-time efficiency. Our research evaluated the YOLO11 architecture against legacy detectors and ultra-lightweight candidates to determine the optimal deployment configuration for clinical environments.
| Model Architecture |mAP50 | Latency (ms) | BackboneR |
|---|---:|---:|---:|
| YOLO11n (Ours) | 0.497 | 10.9 | C3k2/C2PSA |
| NanoDet-m	 | 0.305	 | 6.1 | ShuffleNetV2 |
| Tiny SSD	 | 0.160	 | 18.4	 | VGG16-Lite |

## Explainable AI (XAI) & Clinical Trust
To bridge the gap between "Black Box" algorithms and clinical practitioners, we integrated two distinct interpretability layers. These ensure that the model’s predictions are based on relevant dental morphology rather than image noise.

# Clinical Activation Maps (Saliency)
We generated saliency heatmaps to visualize the model’s "Attention Zones." These maps translate complex neural activations into a color-coded gradient, allowing dentists to verify exactly which anatomical regions—such as the enamel-dentine junction—triggered a positive diagnosis.

# C2PSA Feature Visualizations
We performed a "digital autopsy" by extracting raw activations from the Programmable Sparse Attention (C2PSA) layer.

- Spatial Awareness: Our analysis confirmed that the internal neurons fire specifically on high-contrast biomarkers, such as radiolucencies in the pulp chamber or root angulations.

- Verification: By visualizing these feature maps, we verified the architectural integrity of our pipeline, proving that the model focuses on pathological density changes rather than artifacts in the X-ray film.
## Running Pipelines

The main pipelines are:

```bash
.venv/bin/python pipelines/dual_detector_baseline_ablation.py
.venv/bin/python pipelines/dual_detector_regularisation_ablation.py
.venv/bin/python pipelines/quadrant_detector_ablation.py
.venv/bin/python pipelines/task_specific_detector_fusion.py
.venv/bin/python pipelines/best_checkpoint_test_inference.py
```

The sweep and final training scripts prepare YOLO-formatted data, train the relevant YOLO models, build validation and test predictions, evaluate metrics, and write outputs below `results/<pipeline-name>/`.

`best_checkpoint_test_inference.py` is the inference-only entry point. It loads `models/pathology_best.pt`, `models/tooth_best.pt`, and `models/quadrant_best.pt`, runs the task-specific detector fusion logic on `data/dentex/test_data`, and writes outputs below `results/best_checkpoint_test_inference/predictions/`.

Raw images and annotations stay under `data/dentex/`. Pipeline YOLO workspaces under `results/` write generated labels/configuration and use symlinks to source images rather than copying image data.

The organised result groups are:

```text
results/dual_detector_baseline_ablation/
results/dual_detector_regularisation_ablation/
results/quadrant_detector_ablation/
results/task_specific_detector_fusion/
results/best_checkpoint_test_inference/
```

Training `results.csv` files from YOLO runs were copied under each available `training_curves/` directory.

## What Was Tried

Baseline sweep:

| Experiment | Aggregate AP50 | Aggregate AP | Aggregate AR | Note |
|---|---:|---:|---:|---|
| `baseline_with_tta` | 0.251224 | 0.143236 | 0.230712 | Best main baseline row before assignment tuning |
| `baseline_yolo11n_1024` | 0.234011 | 0.146592 | 0.209901 | YOLO11n baseline |
| `larger_pathology_yolo11s` | 0.224065 | 0.139252 | 0.204252 | Bigger pathology model did not improve aggregate results |
| `low_augmentation` | 0.221626 | 0.140886 | 0.193374 | Reduced augmentation did not help |
| `higher_resolution_1280` | 0.181963 | 0.120001 | 0.171334 | Higher resolution was worse overall |

Assignment tuning on the TTA baseline found nearest assignment to be strongest:

| Assignment | Aggregate AP50 | Aggregate AP | Aggregate AR |
|---|---:|---:|---:|
| `nearest` | 0.290852 | 0.169770 | 0.290883 |
| `iou_first` | 0.269575 | 0.156577 | 0.249705 |
| `hungarian` | 0.259509 | 0.152267 | 0.240579 |

Regularisation sweep:

| Best experiment | Aggregate AP50 | Aggregate AP | Aggregate AR |
|---|---:|---:|---:|
| `lower_lr` | 0.271935 | 0.174615 | 0.280971 |

Quadrant Model C sweep:

| Result | Experiment | Aggregate AP50 |
|---|---|---:|
| Best full Model C pipeline | `quadrant_yolo11s_1024_standard` | 0.460433 |
| Best quadrant-only sanity run | `quadrant_yolo11n_1280_standard` | Quadrant AP50 0.579688 |

## Final Noteworthy Results

The task-specific detector fusion study used:

```text
pathology_model: yolo11n.pt
tooth_model: yolo11n.pt
quadrant_model: yolo11s.pt
image size: 1024
learning rate: 0.003
tooth assignment: nearest
quadrant assignment: nearest
```

Validation metrics:

| Task | AP50 | AP | AR |
|---|---:|---:|---:|
| Quadrant | 0.533471 | 0.331713 | 0.563137 |
| Enumeration | 0.337465 | 0.194060 | 0.316972 |
| Diagnosis | 0.480296 | 0.280027 | 0.469199 |
| Aggregate | 0.450411 | 0.268600 | 0.449769 |

Released test metrics:

| Task | AP50 | AP | AR |
|---|---:|---:|---:|
| Quadrant | 0.506958 | 0.300790 | 0.561575 |
| Enumeration | 0.241249 | 0.142426 | 0.300694 |
| Diagnosis | 0.460345 | 0.265687 | 0.425636 |
| Aggregate | 0.402851 | 0.236301 | 0.429302 |

The dedicated quadrant model materially improved the quadrant task and lifted aggregate performance. Diagnosis on released test data should be interpreted with care because released labels include classes outside the four-class diagnosis label space used for training.
