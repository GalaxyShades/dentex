# dentex

Local research workspace for DENTEX segmentation and detection workflows.

## Repository layout

- `DentexSegAndDet/`: copied source code from `https://github.com/xyzlancehe/DentexSegAndDet`
- `dentex_workflow.ipynb`: single notebook workflow with sections for preprocessing, augmentation, training, and testing

## Dataset

The notebook is designed for Kaggle or Google Colab and downloads DENTEX from:

- `https://huggingface.co/datasets/ibrahimhamamci/DENTEX`

No repository-level `/data` directory is committed.

## Local environment

Create and activate the virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Local dependency installation is intentionally kept empty to avoid downloads on slow connections.

For Kaggle or Colab runs, install:

```bash
pip install -r requirements-colab.txt
```

## Git remote

The repository is initialised locally with `origin` set to:

- `git@github.com:GalaxyShadesCat/dentex.git`
