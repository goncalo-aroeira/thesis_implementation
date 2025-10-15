# thesis_implementation

**Transfer single-cell foundation model representations to predict drug resistance in cancer**

Code for my thesis evaluating **bulk RNA-seq**, **single-cell (pseudobulk)**, and **foundation-model embeddings** (scGPT, scFoundation) for **per-drug response prediction** (LN(IC50)) on cancer cell lines.

> **TL;DR**: Strong, well-regularized **bulk** baselines are hard to beat. Pseudobulk rarely surpasses matched bulk, and zero-shot transfer from single-cell foundation models does **not** consistently outperform classical approaches across drugs.

---

## Repository structure

Plain-text (copy-safe) layout:

.
- bulk_state_of_the_art/  # Bulk RNA-seq pipelines, models, CV & evaluation
- envs/                   # Conda environment YAMLs (one per used env)
- maps/                   # ID maps / lookups (genes, models, tissues, etc.)
- scGPT/                  # scGPT embedding extraction + downstream heads
- single_cell/            # scRNA-seq QC -> pseudobulk -> preprocessing
- .gitignore
- README.md

---

## Setup

### Create the Conda environments

This repo includes YAMLs under `envs/`. Create the ones you need (names may vary—use whatever is in `envs/` on your branch):

```bash
# from repo root
ls envs

# create envs (examples; adjust to your files)
conda env create -f envs/base.yml
conda env create -f envs/scgpt.yml
conda env create -f envs/scfoundation.yml

# activate
conda activate <env-name-from-yaml>
