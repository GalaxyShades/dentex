# DENTEX Insights

This note summarises what was tried, what worked, and what remains risky. The main metric used during the experiments was aggregate AP50 across quadrant, enumeration, and diagnosis, with AP and AP75 used as stricter checks.

## Main Findings

The strongest validation result came from task-specific detector fusion: a pathology detector, a tooth enumeration detector, and a dedicated quadrant detector. This lifted validation aggregate AP50 to `0.450411`, compared with `0.290852` for the best two-detector baseline after assignment tuning.

The biggest technical lesson is that quadrant prediction was the original bottleneck. The two-detector baseline had almost zero quadrant AP50 because quadrant was inferred indirectly from tooth layout. A dedicated quadrant model raised quadrant AP50 to roughly `0.53-0.57` in the Model C experiments and made the full pipeline viable.

Diagnosis performance is useful but visually noisy. Low confidence thresholds improve AP-style metrics, but they keep many low-confidence detections. A clean review mode reduced validation predictions from `525` to `275`, but also reduced diagnosis AR from `0.472169` to `0.342833`.

## What Was Tried

### Dual-Detector Baseline

The baseline used a pathology detector plus a tooth detector. Quadrant labels were inferred from tooth layout.

| Experiment | Aggregate AP50 | Aggregate AP | Aggregate AP75 | Aggregate AR | Note |
|---|---:|---:|---:|---:|---|
| `baseline_with_tta` | 0.251224 | 0.143236 | 0.134487 | 0.230712 | Best initial baseline row |
| `baseline_yolo11n_1024` | 0.234011 | 0.146592 | 0.171213 | 0.209901 | Plain YOLO11n baseline |
| `larger_pathology_yolo11s` | 0.224065 | 0.139252 | 0.157448 | 0.204252 | Larger pathology model did not help |
| `low_augmentation` | 0.221626 | 0.140886 | 0.161002 | 0.193374 | Reduced augmentation did not help |
| `higher_resolution_1280` | 0.181963 | 0.120001 | 0.136273 | 0.171334 | Higher resolution was worse |

**Noteworthy:** bigger and higher-resolution models did not improve the baseline. The bottleneck was not simply model capacity or input size.

### Assignment Ablation

The method used to assign pathology boxes to teeth mattered.

| Assignment | Predictions | Aggregate AP50 | Aggregate AP | Aggregate AP75 | Aggregate AR |
|---|---:|---:|---:|---:|---:|
| `nearest` | 645 | 0.290852 | 0.169770 | 0.166716 | 0.290883 |
| `iou_first` | 406 | 0.269575 | 0.156577 | 0.154702 | 0.249705 |
| `hungarian` | 408 | 0.259509 | 0.152267 | 0.150966 | 0.240579 |

**Noteworthy:** nearest assignment produced the best aggregate result, but it also produced more predictions. This is one reason AP-style improvements should be checked against false-positive behaviour.

### Threshold Tuning

The original threshold grid selected:

```text
pathology_conf = 0.15
tooth_conf     = 0.25
predict_iou    = 0.50
```

This was the best baseline threshold setting by aggregate AP50. However, it was not optimised specifically for false-positive control.

**Noteworthy:** stricter confidence settings reduced the number of predictions, but AP50 did not always reward that. Threshold choice should depend on whether the output is intended for scoring or review.

### Regularisation Sweep

Lower learning rate was the strongest regularisation change.

| Experiment | Aggregate AP50 | Aggregate AP | Aggregate AP75 | Aggregate AR | Note |
|---|---:|---:|---:|---:|---|
| `lower_lr` | 0.271935 | 0.174615 | 0.205416 | 0.280971 | Best regularisation row |
| `base_config` | 0.247804 | 0.154777 | 0.174764 | 0.253641 | Reference row |
| `stricter_prediction_confidence` | 0.226512 | 0.139239 | 0.168122 | 0.217927 | Cleaner but lower recall |
| `higher_weight_decay` | 0.221080 | 0.131550 | 0.140731 | 0.244554 | Did not help |
| `shorter_patience` | 0.185841 | 0.109570 | 0.119153 | 0.232473 | Underperformed |

**Noteworthy:** stricter confidence reduced predictions, but it reduced aggregate metrics. This foreshadowed the later precision-versus-recall trade-off.

### Dedicated Quadrant Detector

Adding a third detector for quadrant labels was the strongest modelling change.

| Experiment | Quadrant AP50 | Enumeration AP50 | Diagnosis AP50 | Aggregate AP50 | Aggregate AP75 |
|---|---:|---:|---:|---:|---:|
| `quadrant_yolo11s_1024_standard` | 0.568447 | 0.314442 | 0.498410 | 0.460433 | 0.305288 |
| `quadrant_yolo11n_1280_standard` | 0.551295 | 0.307667 | 0.501338 | 0.453433 | 0.297891 |
| `quadrant_yolo11n_1024_standard` | 0.529468 | 0.311961 | 0.485017 | 0.442149 | 0.288898 |
| `quadrant_yolo11n_1024_low_aug` | 0.488075 | 0.313835 | 0.496383 | 0.432764 | 0.282547 |

**Noteworthy:** the dedicated quadrant detector changed the character of the pipeline. The quadrant task moved from near-zero AP50 in the baseline to the strongest subtask in the fused model.

## Final Fusion Result

The final task-specific detector fusion used:

```text
pathology_model: yolo11n.pt
tooth_model: yolo11n.pt
quadrant_model: yolo11s.pt
image size: 1024
learning rate: 0.003
tooth assignment: nearest
quadrant assignment: nearest
```

Validation:

| Task | AP50 | AP | AP75 | AR |
|---|---:|---:|---:|---:|
| Quadrant | 0.533471 | 0.331713 | 0.390408 | 0.563137 |
| Enumeration | 0.337465 | 0.194060 | 0.183727 | 0.316972 |
| Diagnosis | 0.480296 | 0.280027 | 0.284100 | 0.469199 |
| Aggregate | 0.450411 | 0.268600 | 0.286078 | 0.449769 |

Released test:

| Task | AP50 | AP | AP75 | AR |
|---|---:|---:|---:|---:|
| Quadrant | 0.506958 | 0.300790 | 0.339354 | 0.561575 |
| Enumeration | 0.241249 | 0.142426 | 0.158113 | 0.300694 |
| Diagnosis | 0.460345 | 0.265687 | 0.290705 | 0.425636 |
| Aggregate | 0.402851 | 0.236301 | 0.262724 | 0.429302 |

**Noteworthy:** released-test aggregate AP50 was lower than validation, mainly because enumeration dropped. Quadrant remained relatively strong.

## AP75 Threshold Sweep

An AP75-focused threshold sweep selected:

```text
pathology_conf = 0.15
tooth_conf     = 0.15
quadrant_conf  = 0.15
aggregate_AP75 = 0.290429
```

This setting improved validation AP75 slightly over the final fusion setting (`0.290429` versus `0.286078`), but it did not solve the visual false-positive issue.

**Noteworthy:** optimising AP75 still preferred a low pathology threshold. AP75 is stricter on localisation, but it is still an AP metric and can still favour recall-heavy ranked detections.

## Clean Review Mode

A clean review mode was tested for visual inspection:

```text
pathology_conf = 0.30
tooth_conf = 0.15
quadrant_conf = 0.15
class-agnostic pathology NMS IoU = 0.50
max pathology detections per assigned tooth = 1
```

Validation comparison:

| Mode | Predictions | Aggregate AP50 | Aggregate AP75 | Aggregate AR | Diagnosis AP75 | Diagnosis AR |
|---|---:|---:|---:|---:|---:|---:|
| Standard | 525 | 0.447491 | 0.290429 | 0.454792 | 0.284455 | 0.472169 |
| Clean review | 275 | 0.396510 | 0.260650 | 0.353664 | 0.242038 | 0.342833 |

**Noteworthy:** clean review mode cut predictions by about 48%, but it reduced recall. It is suitable for cleaner qualitative inspection, not as a replacement for the score-optimised submission path.

## Released Test Label Caveat

Released test labels use a broader diagnosis label space than the four diagnosis classes used for training.

Trained diagnosis classes:

- `0`: impacted
- `1`: caries
- `2`: periapical lesion
- `3`: deep caries

Comparable released-test labels:

- `1-çürük` -> `caries`
- `6-gömülü` -> `impacted`
- `7-lezyon` -> `periapical_lesion`

Other released-test labels such as healthy teeth, root canal, extraction, fracture, and curettage are outside the trained diagnosis label space.

**Noteworthy:** released-test diagnosis metrics should be reported only after filtering or mapping labels to the trained label space. Otherwise the evaluation is not an apples-to-apples measurement of the trained model.

## Practical Conclusion

The project moved from a weak layout-derived quadrant baseline to a much stronger task-specific detector fusion system. The dedicated quadrant detector was the decisive improvement. The remaining weakness is not only model accuracy; it is also operating-point selection. Low thresholds support AP-style scoring, while clean clinical review needs stricter post-processing and accepts lower recall.
