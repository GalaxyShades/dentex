# DENTEX Configuration Findings

This report summarises configuration trends from the exported results in `dentex_results/dentex_baseline_standalone/predictions`. The main comparison table is `experiment_comparison.csv`; assignment behaviour is taken from `assignment_ablation.csv`.

## Summary

The strongest current configuration is the YOLO11n 1024 baseline with TTA and nearest assignment. The evidence does not support moving to a larger pathology model, 1280-pixel training, or low augmentation. The most useful changes in the completed runs were TTA and nearest assignment.

| Finding | Supported conclusion |
|---|---|
| Bigger model did not help | YOLO11s pathology reduced aggregate AP50 versus the YOLO11n baseline. |
| 1280 resolution did not help | 1280 had the worst aggregate AP50 among the main experiments. |
| Low augmentation did not help | Low augmentation reduced aggregate AP50, diagnosis AP50, and AR. |
| TTA helped | TTA improved aggregate AP50, diagnosis AP50, enumeration AP50, and AR. |
| Nearest assignment helped | Nearest assignment produced the best aggregate AP50, AP, and AR. |

## Main Experiment Results

| Experiment | Predictions | Quadrant AP50 | Enumeration AP50 | Diagnosis AP50 | Aggregate AP50 | Aggregate AP | Aggregate AR |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_yolo11n_1024 | 194 | 0.001292 | 0.296676 | 0.404065 | 0.234011 | 0.146592 | 0.209901 |
| baseline_with_tta | 364 | 0.001484 | 0.327134 | 0.425054 | 0.251224 | 0.143236 | 0.230712 |
| larger_pathology_yolo11s | 212 | 0.002171 | 0.248810 | 0.421214 | 0.224065 | 0.139252 | 0.204252 |
| low_augmentation | 185 | 0.001482 | 0.292155 | 0.371240 | 0.221626 | 0.140886 | 0.193374 |
| higher_resolution_1280 | 161 | 0.003307 | 0.213896 | 0.328686 | 0.181963 | 0.120001 | 0.171334 |

## Bigger Model Did Not Help

The larger pathology model experiment used `yolo11s.pt` for pathology while keeping the tooth model at `yolo11n.pt`. Compared with the `baseline_yolo11n_1024` run, it reduced the main aggregate metrics:

| Metric | YOLO11n baseline | Larger pathology YOLO11s | Change |
|---|---:|---:|---:|
| Aggregate AP50 | 0.234011 | 0.224065 | -0.009946 |
| Aggregate AP | 0.146592 | 0.139252 | -0.007340 |
| Aggregate AR | 0.209901 | 0.204252 | -0.005649 |
| Enumeration AP50 | 0.296676 | 0.248810 | -0.047866 |

Diagnosis AP50 increased slightly, from `0.404065` to `0.421214`, but that gain was not enough to offset worse enumeration and aggregate performance. For the full DENTEX pipeline, the larger model did not help.

## 1280 Resolution Did Not Help

The `higher_resolution_1280` run was worse than the 1024 baseline on all aggregate metrics:

| Metric | 1024 baseline | 1280 run | Change |
|---|---:|---:|---:|
| Aggregate AP50 | 0.234011 | 0.181963 | -0.052048 |
| Aggregate AP | 0.146592 | 0.120001 | -0.026591 |
| Aggregate AR | 0.209901 | 0.171334 | -0.038567 |
| Diagnosis AP50 | 0.404065 | 0.328686 | -0.075380 |
| Enumeration AP50 | 0.296676 | 0.213896 | -0.082780 |

Quadrant AP50 increased slightly, but quadrant AP50 is near zero in all runs, so this does not represent a useful practical improvement. The 1280 configuration was the weakest main experiment by aggregate AP50.

## Low Augmentation Did Not Help

The low-augmentation run reduced mosaic, scale, and rotation. It underperformed the 1024 baseline:

| Metric | 1024 baseline | Low augmentation | Change |
|---|---:|---:|---:|
| Aggregate AP50 | 0.234011 | 0.221626 | -0.012385 |
| Aggregate AP | 0.146592 | 0.140886 | -0.005706 |
| Aggregate AR | 0.209901 | 0.193374 | -0.016526 |
| Diagnosis AP50 | 0.404065 | 0.371240 | -0.032826 |
| Enumeration AP50 | 0.296676 | 0.292155 | -0.004521 |

This suggests that simply reducing augmentation is not a good anti-overfitting strategy for this setup. The current model appears to need at least moderate augmentation.

## TTA Helped

TTA improved the main practical metrics relative to the same YOLO11n 1024 baseline without TTA:

| Metric | No TTA baseline | TTA run | Change |
|---|---:|---:|---:|
| Predictions | 194 | 364 | +170 |
| Aggregate AP50 | 0.234011 | 0.251224 | +0.017213 |
| Aggregate AR | 0.209901 | 0.230712 | +0.020811 |
| Diagnosis AP50 | 0.404065 | 0.425054 | +0.020989 |
| Enumeration AP50 | 0.296676 | 0.327134 | +0.030458 |

Aggregate AP dropped slightly from `0.146592` to `0.143236`, but AP50 and AR improved. Since the target pipeline currently benefits from more recovered detections, TTA is useful.

## Nearest Assignment Helped

The assignment ablation was run on the TTA configuration. Nearest assignment gave the best aggregate AP50, AP, and AR:

| Assignment | Predictions | Quadrant AP50 | Enumeration AP50 | Diagnosis AP50 | Aggregate AP50 | Aggregate AP | Aggregate AR |
|---|---:|---:|---:|---:|---:|---:|---:|
| hungarian | 408 | 0.001499 | 0.349950 | 0.427077 | 0.259509 | 0.152267 | 0.240579 |
| iou_first | 406 | 0.001287 | 0.334245 | 0.473193 | 0.269575 | 0.156577 | 0.249705 |
| nearest | 645 | 0.001357 | 0.357106 | 0.514094 | 0.290852 | 0.169770 | 0.290883 |

Compared with Hungarian assignment, nearest assignment changed the metrics as follows:

| Metric | Hungarian | Nearest | Change |
|---|---:|---:|---:|
| Predictions | 408 | 645 | +237 |
| Aggregate AP50 | 0.259509 | 0.290852 | +0.031344 |
| Aggregate AP | 0.152267 | 0.169770 | +0.017503 |
| Aggregate AR | 0.240579 | 0.290883 | +0.050304 |
| Diagnosis AP50 | 0.427077 | 0.514094 | +0.087016 |
| Enumeration AP50 | 0.349950 | 0.357106 | +0.007156 |

Nearest assignment slightly reduced quadrant AP50, but quadrant AP50 is effectively zero for all assignment modes. The aggregate gain is mainly from better diagnosis and recall.

## Practical Recommendation

The current evidence supports using the best existing configuration as the base rather than searching through many broad model configurations:

- `yolo11n.pt`
- `imgsz=1024`
- moderate augmentation
- TTA enabled
- nearest assignment

Further work should focus on targeted generalisation changes and the separate Model C quadrant detector experiment, not on larger models or higher resolution.
