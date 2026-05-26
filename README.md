# CNeuroMod GLM

Applies GLM to CNeuroMod fMRI data to compute task-based contrasts and effect maps across subjects.

The pipeline runs session-level first-level GLMs (nilearn), aggregates them into subject-level
fixed-effects maps, extracts Fedorenko language fROIs (top-voxel selection within parcels),
and produces summary visualizations.

---

## Quick Start

```bash
uv sync
uv run invoke fetch
uv run invoke run
```

---

## Setup

```bash
uv sync
```

This creates a `.venv` and installs all dependencies from `pyproject.toml`.

---

## Usage

### Fetch source data

```bash
uv run invoke fetch
```

Downloads the Fedorenko atlas and installs the CNeuroMod DataLad superdataset.
Requires SSH access to `git@github.com:courtois-neuromod/cneuromod.all.git`.

To fetch data for a specific dataset or subset of subjects/tasks:

```bash
uv run invoke fetch --dataset langlocalizer
uv run invoke fetch --dataset hcptrt --subjects sub-01,sub-02 --tasks motor,language
```

### Run the full pipeline

```bash
uv run invoke run
```

Steps that already have outputs are skipped. To force a full rerun:

```bash
uv run invoke clean
uv run invoke run
```

### Run individual steps

```bash
uv run invoke run-glm              # Session-level first-level GLM
uv run invoke run-subject          # Subject-level fixed-effects averaging
uv run invoke run-froi             # Fedorenko fROI extraction
uv run invoke run-notebooks        # Visualization figures
```

Each step supports `--dataset`, `--subjects`, and `--tasks` for partial runs:

```bash
uv run invoke run-glm --dataset langlocalizer --subjects sub-01 --tasks listening
uv run invoke run-glm --dataset hcptrt --subjects sub-01 --tasks motor,language
```

### Smoke test

```bash
uv run invoke run-smoke
```

Runs the full pipeline on the first subject only to verify end-to-end wiring.

### Clean outputs

```bash
uv run invoke clean               # Remove all computed outputs
uv run invoke clean-glm           # Remove session-level GLM maps only
uv run invoke clean-subject       # Remove subject-level maps only
uv run invoke clean-froi          # Remove fROI masks only
uv run invoke clean-source        # Remove downloaded source data (atlas)
```

---

## Task Overview

| Task             | Description                                                   |
| ---------------- | ------------------------------------------------------------- |
| `fetch`          | Download Fedorenko atlas; clone & get CNeuroMod DataLad data  |
| `run-glm`        | Session-level first-level GLM (nilearn); skip if outputs exist |
| `run-subject`    | Fixed-effects subject-level averaging; skip if outputs exist  |
| `run-froi`       | Fedorenko language fROI extraction; skip if outputs exist     |
| `run-notebooks`  | Visualization figures (reads from `output_data/`)             |
| `run`            | Full pipeline in order                                        |
| `run-smoke`      | Minimal end-to-end test (first subject only)                  |
| `clean-glm`      | Remove session-level GLM outputs                              |
| `clean-subject`  | Remove subject-level outputs                                  |
| `clean-froi`     | Remove fROI mask outputs                                      |
| `clean`          | Remove all computed outputs                                   |
| `clean-source`   | Remove downloaded source data                                 |

Use `uv run invoke --list` for all tasks.

---

## Configuration

Edit `invoke.yaml` to change defaults:

- `datasets` — dict of datasets; each entry has `model`, `subjects`, and `task_names`.
  Currently configured: `langlocalizer` and `hcptrt`.
- `tr`, `smoothing_fwhm`, `n_compcor` — GLM parameters (shared across datasets)
- `froi_top_percent` — fraction of top voxels selected per Fedorenko parcel

Per-dataset scientific specifications (contrasts, event transformations, file patterns)
live in `models/{dataset}.json` — BIDS Stats Models 1.0.0 format.

---

## Folder Structure

| Folder / File       | Description                                          |
| ------------------- | ---------------------------------------------------- |
| `analysis/`         | Pure Python analysis code (GLM, fROI functions, model spec parser) |
| `models/`           | BIDS Stats Models 1.0.0 descriptors — one per dataset |
| `notebooks/`        | Visualization notebooks (read from `output_data/`)   |
| `source_data/`      | Raw inputs — see [`source_data/CONTENT.md`](source_data/CONTENT.md) |
| `output_data/`      | Results and figures — see [`output_data/CONTENT.md`](output_data/CONTENT.md) |
| `tasks.py`          | Invoke task definitions                              |
| `invoke.yaml`       | Config: datasets, paths, GLM parameters              |

---

## Atlas License Note

The Fedorenko language parcellation atlas (`allParcels-language-SN220`) is downloaded from
MIT EvLab servers. No explicit license is currently provided by the authors.
A properly licensed deposit on Zenodo is planned to address this reproducibility concern.
Reference: Fedorenko et al. (2010), *Journal of Neurophysiology* 104(2): 1177–1194.
