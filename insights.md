# Insights

## Released Test Diagnosis Labels

The final three-model pipeline performed much better than the original baseline on released test data overall, mainly because the dedicated quadrant model fixed the quadrant task.

However, diagnosis still looked poor on released test data. Further inspection showed that the released test labels use a broader diagnosis label space than the model was trained on.

The model was trained on four diagnosis classes:

- `0`: impacted
- `1`: caries
- `2`: periapical lesion
- `3`: deep caries

The released test LabelMe files contained labels such as:

- `0-saglam`: healthy / sound
- `1-çürük`: caries
- `2-küretaj`: curettage / periodontal treatment-related label
- `3-kanal`: root canal / endodontic treatment
- `5-çekim`: extraction
- `6-gömülü`: impacted
- `7-lezyon`: lesion, likely periapical lesion
- `8-kırık`: fractured / broken

The comparable released-test labels are:

- `1-çürük` -> model class `1` caries
- `6-gömülü` -> model class `0` impacted
- `7-lezyon` -> model class `2` periapical lesion

The other released-test labels are outside the model's trained diagnosis classes and should be omitted from released-test metric calculation when reporting performance against the trained label space.

This means the original released-test diagnosis score was not an apples-to-apples evaluation of the trained four-class diagnosis model.
