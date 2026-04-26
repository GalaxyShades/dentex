# BMED 4507 Final Presentation Plan

Project title: **Automated Dental Disease Detection and Tooth Labelling in Panoramic X-rays**

Target length: **8 minutes**

Audience assumption: most classmates understand biomedical imaging at a high level, but may not know object detection metrics or dental numbering systems. Keep the story visual, explain terms plainly, and avoid spending time on implementation details unless they explain a design choice.

## Core Message

Dental panoramic X-rays contain many teeth and several possible abnormalities. This project built and tested a deep learning pipeline that detects diseased teeth and predicts three clinically useful labels:

- **Where in the mouth?** Quadrant.
- **Which tooth?** Enumeration, meaning tooth position within the quadrant.
- **What disease?** Diagnosis, such as caries, deep caries, impacted tooth, or periapical lesion.

The main lesson is that disease detection alone was not enough. The strongest pipeline came from splitting the problem into specialised detectors: one for pathology, one for tooth enumeration, and one dedicated to quadrant labelling.

## Recommended Slide Flow

Use **8 slides**. Aim for roughly **1 minute per slide**, with the final slide slightly shorter.

### Slide 1: Title and One-Sentence Summary

**Title:** Automated Dental Disease Detection and Tooth Labelling in Panoramic X-rays

**On-slide content:**

- Biomedical task: object detection in dental panoramic X-rays.
- Output per finding: bounding box + quadrant + tooth number + diagnosis.
- Final released-test aggregate: **AP50 = 0.403**.

**Visual design:**

- Full-width dental panoramic X-ray example as the background or main visual.
- Add 2 to 3 coloured boxes on teeth with simple labels, for example `Q2 / Tooth 6 / Caries`.
- Use one accent colour for boxes and one neutral colour for text.

**Speaker notes:**

> This project is about helping interpret dental panoramic X-rays. Instead of only saying whether an abnormality exists, the model tries to localise it and attach clinically useful labels: the mouth quadrant, tooth position, and disease type.

### Slide 2: Clinical Motivation

**Title:** Why This Matters

**On-slide content:**

- Dental X-rays are widely used for screening and treatment planning.
- Manual review can be time-consuming and depends on reader experience.
- Automated detection can support triage, second reading, and structured reporting.
- The task is difficult because teeth overlap, diseases can be subtle, and labels require anatomical context.

**Visual design:**

- Left: panoramic X-ray with dense tooth structures.
- Right: three simple icons or tags: `Locate`, `Number`, `Diagnose`.
- Keep text to four short bullets.

**Speaker notes:**

> The clinical motivation is decision support, not replacing dentists. A useful system must do more than draw boxes. It also needs to say which tooth is affected and what abnormality is likely present.

### Slide 3: Dataset and Prediction Task

**Title:** Dataset and Labels

**On-slide content:**

| Split | Images | Notes |
|---|---:|---|
| Training | 705 | Used to train YOLO detectors |
| Validation | 50 | Used for model selection and analysis |
| Test | 250 | Used for released-test evaluation/export |

Each labelled finding includes:

- Bounding box around the tooth/pathology region.
- Quadrant: mouth region.
- Enumeration: tooth position within the quadrant.
- Diagnosis: pathology class.

**Visual design:**

- Use a simple diagram: `Image -> Box -> Quadrant + Enumeration + Diagnosis`.
- Include a tiny tooth numbering illustration if available.

**Speaker notes:**

> Enumeration means tooth position within a quadrant, usually from 1 to 8. This matters because the same disease label is less useful if the system cannot say which tooth it belongs to.

### Slide 4: Method Overview

**Title:** Three-Model Pipeline

**On-slide content:**

Final pipeline:

1. **Model A:** detects pathology and predicts diagnosis.
2. **Model B:** detects teeth and predicts enumeration.
3. **Model C:** detects teeth/quadrants and predicts quadrant.
4. Labels are assigned to pathology boxes using nearest spatial matching.

Final configuration:

- YOLO11n for pathology.
- YOLO11n for tooth enumeration.
- YOLO11s for quadrant.
- Image size: 1024.
- Lower learning rate: 0.003.

**Visual design:**

Use a horizontal pipeline:

`Panoramic X-ray -> Pathology detector -> Tooth detector -> Quadrant detector -> Combined prediction`

Show each detector as a small labelled block, not as code.

**Speaker notes:**

> The pipeline separates the problem into parts. One model focuses on abnormality and disease class, another focuses on tooth identity, and a third focuses on quadrant. This was motivated by early experiments where quadrant labelling was the weakest part.

### Slide 5: Experimental Design

**Title:** How The Model Was Chosen

**On-slide content:**

Experiments across notebooks:

- **Baseline:** pathology + tooth enumeration models.
- **Regularisation sweep:** learning rate, patience, weight decay, confidence thresholds.
- **Quadrant experiment:** added a dedicated Model C for quadrant prediction.
- **Final notebook:** combined lower learning rate with the best quadrant model.

Key validation comparison:

| Experiment stage | Aggregate AP50 | Main finding |
|---|---:|---|
| Baseline A+B | 0.291 | Diagnosis and enumeration worked; quadrant failed |
| Lower learning rate A+B | 0.272 | Improved diagnosis but still weak quadrant |
| A+B+C quadrant pipeline | 0.460 | Dedicated quadrant model fixed major bottleneck |
| Final model | 0.450 | Similar validation performance, then tested on release labels |

**Visual design:**

- A simple bar chart of aggregate AP50 by stage.
- Highlight the jump after adding Model C.

**Speaker notes:**

> The important experimental result is not just the final score. It is the diagnosis of the pipeline failure. The baseline could detect disease and tooth position reasonably, but quadrant AP50 was almost zero. Adding a separate quadrant model made the biggest improvement.

### Slide 6: Final Results

**Title:** Released-Test Results

**On-slide content:**

| Task | AP | AP50 | AP75 | AR |
|---|---:|---:|---:|---:|
| Quadrant | 0.301 | 0.507 | 0.339 | 0.562 |
| Enumeration | 0.142 | 0.241 | 0.158 | 0.301 |
| Diagnosis | 0.266 | 0.460 | 0.291 | 0.426 |
| **Aggregate** | **0.236** | **0.403** | **0.263** | **0.429** |

Plain-language interpretation:

- Quadrant prediction was strongest by AP50.
- Diagnosis was moderate.
- Enumeration was the main remaining weakness.

**Visual design:**

- Use a grouped bar chart for AP50 only: Quadrant, Enumeration, Diagnosis.
- Put the full table in small text or appendix if the slide feels crowded.

**Speaker notes:**

> AP50 means the predicted box overlaps the ground truth by at least 50 percent and has the correct label. On the released test set, quadrant AP50 was about 0.51, diagnosis AP50 was about 0.46, and enumeration AP50 was only about 0.24. So the model often found the region, but exact tooth numbering remained hard.

### Slide 7: Error Analysis

**Title:** What Went Wrong?

**On-slide content:**

Validation error decomposition:

| Error type | Count |
|---|---:|
| False positive | 339 |
| Correct | 101 |
| False negative or localisation failure | 25 |
| Enumeration error | 24 |
| Diagnosis error | 23 |
| Multiple label error | 6 |
| Quadrant error | 3 |

Main interpretation:

- Too many extra predictions were produced.
- Tooth enumeration and diagnosis errors were similar in count.
- Quadrant errors became rare after adding Model C.

**Visual design:**

- Horizontal bar chart of error counts.
- Use grey for false positives and an accent colour for the main label errors.

**Speaker notes:**

> The most common error was false positives, meaning the model predicted extra findings that did not match labelled disease regions. Among matched boxes, enumeration and diagnosis were the next major issues. Quadrant errors were low, which supports the decision to add a dedicated quadrant model.

### Slide 8: Discussion, Limitations, and Takeaways

**Title:** Lessons Learned

**On-slide content:**

What worked:

- Decomposing the task into specialised detectors.
- Official-style AP, AP50, AP75, and AR metrics.
- Validation error analysis beyond headline scores.

Limitations:

- Small validation set: 50 images.
- Enumeration remained difficult.
- False positives were high.
- YOLO boxes do not explicitly model dental anatomy or tooth ordering.

Future work:

- Add anatomy-aware post-processing for tooth order.
- Improve confidence calibration to reduce false positives.
- Use segmentation or keypoints to better separate adjacent teeth.

**Visual design:**

- Three columns: `Worked`, `Limitations`, `Next steps`.
- End with one final sentence at the bottom: **Specialised anatomical labelling matters as much as disease detection.**

**Speaker notes:**

> The final takeaway is that a medically useful dental model needs localisation, disease recognition, and anatomical labelling. The experiments showed that these are different sources of difficulty. Future work should use dental structure more directly, especially for tooth numbering and false-positive control.

## 8-Minute Timing Plan

| Slide | Time | Focus |
|---|---:|---|
| 1 | 0:40 | Project goal |
| 2 | 0:55 | Biomedical motivation |
| 3 | 1:00 | Dataset and labels |
| 4 | 1:15 | Method |
| 5 | 1:20 | Experimental design |
| 6 | 1:20 | Results |
| 7 | 1:00 | Error analysis |
| 8 | 0:50 | Limitations and takeaways |

Total: **8:20**. During rehearsal, shorten Slide 5 or Slide 6 by 20 seconds.

## Suggested Opening Script

> Our project focuses on dental panoramic X-rays. The goal is to detect abnormal teeth and attach three labels to each prediction: where it is in the mouth, which tooth it is, and what disease is present. This is a biomedical object detection problem, but it also requires anatomical labelling, which turned out to be one of the hardest parts.

## Suggested Results Script

> The final released-test aggregate AP50 was 0.403. Breaking this down, quadrant AP50 was 0.507, diagnosis AP50 was 0.460, and enumeration AP50 was 0.241. This shows that the model was much better at assigning broad mouth region and disease class than exact tooth position. The validation error analysis also showed many false positives, so future work should focus on confidence calibration and anatomy-aware post-processing.

## Suggested Closing Script

> The main conclusion is that dental X-ray analysis is not only a disease detection problem. For a prediction to be clinically useful, it must also identify the correct tooth. Our best improvement came from recognising this and adding a dedicated quadrant detector. The remaining challenge is finer anatomical reasoning, especially enumeration and false-positive reduction.

## Slide Design Guidelines

- Use a clean clinical style: white or very light grey background, dark text, one blue or teal accent colour.
- Put X-ray images at the centre of the story. Avoid code screenshots.
- Use AP50 for most charts because it is easiest to explain in 8 minutes.
- Keep tables small. Round metrics to three decimals.
- Avoid explaining every YOLO hyperparameter. Mention only choices that affected the result.
- Use consistent colours:
  - Quadrant: blue.
  - Enumeration: orange.
  - Diagnosis: green.
  - False positives/errors: grey or red.
- Use the same three words throughout: **Locate, Number, Diagnose**.

## Figures To Prepare

1. Example panoramic X-ray with predicted boxes and labels.
2. Pipeline diagram showing Models A, B, and C.
3. Validation aggregate AP50 bar chart across experiment stages.
4. Released-test AP50 bar chart by task.
5. Error decomposition horizontal bar chart.

## Appendix Material

Use these only if asked during Q&A.

### Full Validation Metrics

| Task | AP | AP50 | AP75 | AR |
|---|---:|---:|---:|---:|
| Quadrant | 0.332 | 0.533 | 0.390 | 0.563 |
| Enumeration | 0.194 | 0.337 | 0.184 | 0.317 |
| Diagnosis | 0.280 | 0.480 | 0.284 | 0.469 |
| **Aggregate** | **0.269** | **0.450** | **0.286** | **0.450** |

### Per-Class Validation AP50

Diagnosis:

| Class | AP50 |
|---|---:|
| Impacted | 0.793 |
| Caries | 0.341 |
| Periapical lesion | 0.355 |
| Deep caries | 0.431 |

Enumeration:

| Tooth position | AP50 |
|---|---:|
| 1 | 0.406 |
| 2 | 0.168 |
| 3 | 0.000 |
| 4 | 0.346 |
| 5 | 0.262 |
| 6 | 0.510 |
| 7 | 0.301 |
| 8 | 0.706 |

Quadrant:

| Quadrant | AP50 |
|---|---:|
| 1 | 0.491 |
| 2 | 0.470 |
| 3 | 0.573 |
| 4 | 0.599 |

### Code and Reproducibility Summary

Submitted notebooks:

- `dentex.ipynb`: baseline pipeline, official-style metrics, threshold tuning, assignment ablation, oracle analysis, bootstrap intervals.
- `regularisation.ipynb`: targeted sweep over training and prediction regularisation.
- `quadrant.ipynb`: dedicated quadrant detector experiment.
- `final.ipynb`: final training, validation evaluation, released-test evaluation, and download bundle creation.

Final artefacts:

- `final_results/predictions/final_summary.json`
- `final_results/predictions/final_metrics_validation_official_style.json`
- `final_results/predictions/final_metrics_released_test_official_style.json`
- `final_results/predictions/predictions_challenge_format_test.json`
- `final_results/download_bundle.zip`

Environment:

- Kaggle runtime.
- Ultralytics YOLO `8.3.32`.
- PyTorch with CUDA on Tesla T4.
- Main Python dependencies installed in the notebook.

