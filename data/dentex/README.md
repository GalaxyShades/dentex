# DENTEX Data Directory

Place the local DENTEX dataset in this directory before running the pipelines.

Expected layout:

```text
data/dentex/
  training_data/
  validation_data/
  test_data/
  validation_triple.json
```

The pipelines read this directory directly and do not call Kaggle APIs.
