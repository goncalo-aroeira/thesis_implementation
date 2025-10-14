# Tahoe → Bulk transfer-learning + perturbation interpretation

This repo-less bundle contains a **single Python script** that runs your full pipeline:

- Train bulk models on pseudo-bulk embeddings (PCs + tissue + growth rate)
- Predict baseline response on **Tahoe DMSO_TF** control cells
- Compute treatment shifts (ΔPC) and test whether they align with the model’s predicted sensitivity/resistance direction
- (Optional) Back-project model coefficients to gene space if you provide bulk PCA loadings

## Files

- `tahoe_transfer_pipeline.py` — the end-to-end pipeline script.
- `config_transfer_pipeline.json` — edit this with your paths and options (see below).

## How to run

1. Edit `config_transfer_pipeline.json` with your actual paths:
   ```json
   {
     "tahoe_parquet": "/path/to/tahoe.parquet",
     "bulk_table_path": "/path/to/bulk_training.parquet",
     "output_dir": "outputs_transfer_pipeline",
     "artifacts_dir": "artifacts_transfer_pipeline",
     "dmso_label": "DMSO_TF",
     "n_pcs": 30,
     "do_procrustes_alignment": true,
     "restrict_to_drugs": ["5-Azacytidine", "5-Fluorouracil", "Oxaliplatin"]
   }
   ```
2. (Optional) Set `TRANSFER_PIPELINE_CONFIG=/path/to/config_transfer_pipeline.json` or just place the JSON next to the script.
3. Run:
   ```bash
   python tahoe_transfer_pipeline.py
   ```

The script will:
- Align Tahoe PCs to bulk PCs using overlapping **SANGER_MODEL_ID** (Procrustes), **only if** ≥5 overlaps exist.
- Standardize features using a scaler fit **on bulk**.
- Train per-drug ElasticNet models (unless disabled) and save them under `artifacts_transfer_pipeline/`.
- Predict on Tahoe controls and export plots + CSVs.
- For each treated drug, compute ΔPC per line and project onto model β (PC part) to derive an **effect score**.
- Correlate baseline predictions vs. effect scores across lines.

## Expected inputs

**Tahoe parquet** (log-normalized expression already; you’ve also added PCs & meta):
```
['drug', 'SANGER_MODEL_ID', 'PC1'...'PC30', 'Tissue_*', 'day4_day1_ratio']
```

**Bulk table** (pseudo-bulk / cell line bulk with assay outcome):
```
['DRUG_ID', 'SANGER_MODEL_ID', 'LN_IC50', 'PC1'...'PC30', 'Tissue_*', 'day4_day1_ratio']
```

> Ensure the **same feature names** exist in both. If Tahoe is missing any `Tissue_*` columns present in bulk, add zero-filled columns before running.

## Outputs (in `outputs_transfer_pipeline/`)

- `tahoe_controls_predictions.csv` — long-form predictions for every Tahoe control cell x drug model.
- `pred_distributions_by_drug.png` — boxplot of control predictions per model.
- `delta_pc_by_line_and_drug.csv` — ΔPC (treated - control) per line & treated drug.
- `effects_baseline_vs_delta.csv` — per line & treated drug, the **effect score** (ΔPC ⋅ β) and the corresponding baseline prediction.
- `baseline_vs_delta_correlations.csv` — Spearman/Pearson correlations between baseline predictions and effect scores, grouped by (drug_model, treated_drug).
- `scatter_<drug_model>_vs_<treated_drug>.png` — top associations plotted as scatterplots.
- (Optional) `beta_genes_long.csv` — model β back-projected to genes if you provide bulk PCA loadings.

## Notes & tips

- **Scale mismatch?** The script runs an **orthogonal Procrustes alignment** from Tahoe PCs to the bulk PC basis using your overlapping SANGER IDs and **controls only**, minimizing treatment-induced bias.
- **Standardization matters.** We apply the **bulk scaler** to both bulk and Tahoe so the model sees comparable feature scales.
- **Interpretation sign.** β > 0 on a PC means increasing that PC raises predicted LN_IC50 (i.e., more resistant). The **effect score** (ΔPC ⋅ β) > 0 implies the treatment pushed the line toward resistance, per the model.
- **Stratify.** You can set `restrict_to_drugs` in config to narrow the analysis to drugs of interest (e.g., `["5-Azacytidine", "5-Fluorouracil", "Oxaliplatin"]`). The script will still compute cross-drug effects (e.g., model=Aza vs treated=5-FU).
- **Back-projection to genes.** Provide `bulk_pca_loadings_path` (genes × PCs with columns `PC1..PCn` and a `gene_id` column) to get gene-level β. You can then intersect with pathway lists for enrichment tests.

## Minimal config template

A starter `config_transfer_pipeline.json` is included; edit paths before running.

## Dependencies

- Python ≥ 3.9
- pandas, numpy, scikit-learn, joblib, matplotlib, scipy

