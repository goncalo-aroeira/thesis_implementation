#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tahoe → Bulk transfer-learning + perturbation interpretation pipeline
====================================================================

What this script does (end-to-end):
1) Load bulk training table and Tahoe single-cell table.
2) (Optional but recommended) Align Tahoe PC space to bulk PC space using overlapping SANGER_MODEL_IDs (orthogonal Procrustes).
3) Standardize Tahoe features using the scaler fitted on the bulk training features (or fit one on bulk here if you're (re)training).
4) Train a drug-specific model on bulk pseudo-bulk embeddings (Elastic Net by default) OR load pre-trained models.
5) Predict "baseline" drug response on Tahoe control cells (drug == 'DMSO_TF').
6) Summarize predictions at single-cell and line-level; export QC plots and tables.
7) For treated cells, compute ΔPC (treated - control) and compare with the direction implied by the model coefficients.
8) Correlate baseline predictions vs. Δ-effect magnitudes, optionally back-project coefficients into gene space if PCA loadings provided.

Inputs (expected columns):
- Tahoe parquet: ['drug', 'SANGER_MODEL_ID', 'PC1'...'PC30', 'Tissue_*', 'day4_day1_ratio', ...]
- Bulk CSV/Parquet: ['DRUG_ID', 'SANGER_MODEL_ID', 'LN_IC50', 'PC1'...'PC30', 'Tissue_*', 'day4_day1_ratio']

Author: ChatGPT (pipeline template)
"""

import os
import json
import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import svd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from joblib import dump, load
import matplotlib.pyplot as plt


# -----------------------------
# Configuration dataclass
# -----------------------------

@dataclass
class Config:
    # Paths
    tahoe_parquet: str = "PATH/TO/tahoe.parquet"
    bulk_table_path: str = "PATH/TO/bulk_training.parquet"  # or .csv
    output_dir: str = "outputs_transfer_pipeline"
    # Columns
    pc_prefix: str = "PC"
    n_pcs: int = 30
    dmso_label: str = "DMSO_TF"
    tissue_prefix: str = "Tissue_"
    growth_col: str = "day4_day1_ratio"
    sanger_col: str = "SANGER_MODEL_ID"
    bulk_drug_col: str = "DRUG_ID"
    bulk_target_col: str = "LN_IC50"
    tahoe_drug_col: str = "drug"
    # Modeling
    random_state: int = 13
    cv_folds: int = 5
    alpha_grid: List[float] = None  # if None, ElasticNetCV default
    l1_ratio_grid: List[float] = None  # if None, ElasticNetCV default
    # Behavior
    train_models_if_missing: bool = True
    do_procrustes_alignment: bool = True
    require_min_overlap_for_alignment: int = 5
    # Optional: bulk PCA loadings to back-project coef into gene space
    bulk_pca_loadings_path: Optional[str] = None  # path to CSV with shape (n_genes, n_pcs) and columns PC1..PCn
    gene_id_col: str = "gene_id"  # only used if bulk_pca_loadings_path is provided
    # Optional: save or load scaler/model
    artifacts_dir: str = "artifacts_transfer_pipeline"
    # Optional: a whitelist of drugs to analyze (strings in bulk DRUG_ID and Tahoe treated labels)
    restrict_to_drugs: Optional[List[str]] = None


# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_pc_columns(cfg: Config) -> List[str]:
    return [f"{cfg.pc_prefix}{i}" for i in range(1, cfg.n_pcs + 1)]


def feature_columns(cfg: Config, df_like: pd.DataFrame) -> List[str]:
    pcs = list_pc_columns(cfg)
    tissues = [c for c in df_like.columns if c.startswith(cfg.tissue_prefix)]
    cols = pcs + tissues
    if cfg.growth_col in df_like.columns:
        cols += [cfg.growth_col]
    return cols


def read_table(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file extension for {path}")


def savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# -----------------------------
# Alignment (Orthogonal Procrustes)
# -----------------------------

def fit_procrustes(X_source: np.ndarray, X_target: np.ndarray) -> np.ndarray:
    """
    Fit an orthogonal matrix R mapping X_source -> X_source @ R ≈ X_target.
    Assumes X_source and X_target are centered (zero mean per column).
    Returns R (n_pcs x n_pcs).
    """
    # Compute cross-covariance
    M = X_source.T @ X_target  # (p x p)
    U, _, Vt = svd(M, full_matrices=False)
    R = U @ Vt  # orthogonal
    return R


def center_columns(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    return X - mu, mu


def align_tahoe_to_bulk(df_tahoe: pd.DataFrame,
                        df_bulk: pd.DataFrame,
                        cfg: Config) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Using overlapping SANGER_MODEL_IDs present in both Tahoe and Bulk, align
    Tahoe PC space to Bulk PC space via orthogonal Procrustes.
    Returns transformed Tahoe df and dict with alignment params for reproducibility.
    """
    pcs = list_pc_columns(cfg)

    # Identify overlap
    overlap_ids = sorted(set(df_tahoe[cfg.sanger_col]).intersection(set(df_bulk[cfg.sanger_col])))
    print(f"[align] Overlap SANGER_MODEL_IDs: {len(overlap_ids)}")

    if len(overlap_ids) < cfg.require_min_overlap_for_alignment:
        print(f"[align] Not enough overlaps ({len(overlap_ids)}) for Procrustes (min={cfg.require_min_overlap_for_alignment}). Skipping alignment.")
        return df_tahoe.copy(), {"used": False}

    # Aggregate Tahoe to line-level means for overlap (use controls to avoid treatment-shift bias)
    tahoe_overlap = (df_tahoe[df_tahoe[cfg.sanger_col].isin(overlap_ids) & (df_tahoe[cfg.tahoe_drug_col] == cfg.dmso_label)]
                     .groupby(cfg.sanger_col)[pcs].mean())

    bulk_overlap = (df_bulk[df_bulk[cfg.sanger_col].isin(overlap_ids)]
                    .groupby(cfg.sanger_col)[pcs].mean())

    # Align matrices by shared order
    tahoe_overlap = tahoe_overlap.loc[overlap_ids]
    bulk_overlap = bulk_overlap.loc[overlap_ids]

    # Center columns
    Xs, mu_s = center_columns(tahoe_overlap.values)
    Xt, mu_t = center_columns(bulk_overlap.values)

    # Fit Procrustes
    R = fit_procrustes(Xs, Xt)

    # Transform all Tahoe PCs: (X - mu_s) @ R + mu_t
    Xtahoe = df_tahoe[pcs].values
    Xtahoe_centered = Xtahoe - mu_s  # broadcast by columns
    Xtahoe_aligned = Xtahoe_centered @ R + mu_t

    df_out = df_tahoe.copy()
    df_out[pcs] = Xtahoe_aligned

    params = {"used": True, "mu_source": mu_s.squeeze().tolist(), "mu_target": mu_t.squeeze().tolist(), "R": R.tolist()}
    return df_out, params


# -----------------------------
# Modeling
# -----------------------------

def fit_scaler_on_bulk(df_bulk: pd.DataFrame, cfg: Config) -> StandardScaler:
    feats = feature_columns(cfg, df_bulk)
    scaler = StandardScaler()
    scaler.fit(df_bulk[feats].values)
    return scaler


def standardize_df_inplace(df: pd.DataFrame, scaler: StandardScaler, cfg: Config):
    feats = feature_columns(cfg, df)
    df.loc[:, feats] = scaler.transform(df[feats].values)


def train_or_load_models(df_bulk: pd.DataFrame, cfg: Config) -> Dict[str, str]:
    """
    Train per-drug ElasticNetCV on bulk and save artifacts.
    Returns dict: drug_id -> path_to_model.joblib
    """
    ensure_dir(cfg.artifacts_dir)
    feats = feature_columns(cfg, df_bulk)
    models = {}

    drugs = sorted(df_bulk[cfg.bulk_drug_col].unique())
    if cfg.restrict_to_drugs:
        drugs = [d for d in drugs if d in set(cfg.restrict_to_drugs)]
    print(f"[train] Training/Loading models for {len(drugs)} drugs.")

    for d in drugs:
        model_path = os.path.join(cfg.artifacts_dir, f"model_{d}.joblib")
        if os.path.exists(model_path) and not cfg.train_models_if_missing:
            models[d] = model_path
            continue

        df_d = df_bulk[df_bulk[cfg.bulk_drug_col] == d].dropna(subset=[cfg.bulk_target_col])
        if df_d.empty:
            print(f"[train] Skipping {d}: no rows with target.")
            continue

        X = df_d[feats].values
        y = df_d[cfg.bulk_target_col].values

        # ElasticNetCV with KFold
        enet = ElasticNetCV(cv=KFold(n_splits=min(cfg.cv_folds, len(df_d)), shuffle=True, random_state=cfg.random_state),
                            l1_ratio=cfg.l1_ratio_grid, alphas=cfg.alpha_grid, random_state=cfg.random_state, n_jobs=None)
        enet.fit(X, y)
        dump(enet, model_path)
        models[d] = model_path

        # Quick CV-like score (on training, not held-out!)
        yhat = enet.predict(X)
        print(f"[train] {d}: R2 (in-sample) = {r2_score(y, yhat):.3f}, n={len(y)}")

    # Save feature list and config for reproducibility
    dump(feats, os.path.join(cfg.artifacts_dir, "features.joblib"))
    with open(os.path.join(cfg.artifacts_dir, "config_used.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)
    return models


def load_models_from_dir(artifacts_dir: str) -> Tuple[Dict[str, object], List[str]]:
    """
    Load all per-drug models and the features list.
    """
    from glob import glob
    model_files = glob(os.path.join(artifacts_dir, "model_*.joblib"))
    models = {}
    for mp in model_files:
        d = os.path.basename(mp).replace("model_", "").replace(".joblib", "")
        models[d] = load(mp)
    feats = load(os.path.join(artifacts_dir, "features.joblib"))
    return models, feats


# -----------------------------
# Inference on Tahoe
# -----------------------------

def predict_on_tahoe_controls(df_tahoe_std: pd.DataFrame, models: Dict[str, object],
                              cfg: Config) -> pd.DataFrame:
    """
    Run all models on Tahoe control cells (DMSO_TF). Returns long-form predictions.
    Columns: ['SANGER_MODEL_ID','drug_model','pred_LN_IC50','cell_idx']
    """
    feats = feature_columns(cfg, df_tahoe_std)
    df_ctrl = df_tahoe_std[df_tahoe_std[cfg.tahoe_drug_col] == cfg.dmso_label].copy()
    if df_ctrl.empty:
        raise ValueError("No Tahoe rows with control label (dmso_label).")

    preds = []
    X = df_ctrl[feats].values
    for d, model in models.items():
        yhat = model.predict(X)
        preds.append(pd.DataFrame({
            cfg.sanger_col: df_ctrl[cfg.sanger_col].values,
            "cell_idx": np.arange(len(df_ctrl)),
            "drug_model": d,
            "pred_LN_IC50": yhat
        }))
    df_pred = pd.concat(preds, ignore_index=True)
    return df_pred


# -----------------------------
# Perturbation analysis (treated vs control)
# -----------------------------

def delta_pc_by_line_and_drug(df_tahoe_aligned: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Compute ΔPC = mean(PC treated) - mean(PC control) for each (SANGER_MODEL_ID, treated_drug).
    Returns wide table with PC columns plus counts.
    """
    pcs = list_pc_columns(cfg)

    # Controls mean per line
    ctrl = (df_tahoe_aligned[df_tahoe_aligned[cfg.tahoe_drug_col] == cfg.dmso_label]
            .groupby(cfg.sanger_col)[pcs].mean()
            .rename(columns={c: f"CTRL_{c}" for c in pcs}))

    treated = df_tahoe_aligned[df_tahoe_aligned[cfg.tahoe_drug_col] != cfg.dmso_label].copy()
    # Drop any weird missing drugs
    treated = treated[treated[cfg.tahoe_drug_col].notna()]

    # Means per (line, drug)
    trt = (treated.groupby([cfg.sanger_col, cfg.tahoe_drug_col])[pcs]
           .mean())

    # Merge to compute deltas
    trt = trt.join(ctrl, on=cfg.sanger_col, how="left")
    for pc in pcs:
        trt[f"DELTA_{pc}"] = trt[pc] - trt[f"CTRL_{pc}"]

    # Also keep counts
    n_ctrl = (df_tahoe_aligned[df_tahoe_aligned[cfg.tahoe_drug_col] == cfg.dmso_label]
              .groupby(cfg.sanger_col).size().rename("n_ctrl"))
    n_trt = treated.groupby([cfg.sanger_col, cfg.tahoe_drug_col]).size().rename("n_treated")

    out = trt.join(n_ctrl, on=cfg.sanger_col, how="left").join(n_trt, how="left")
    out = out.reset_index().rename(columns={cfg.tahoe_drug_col: "treated_drug"})
    return out


def beta_projection_and_correlation(df_delta: pd.DataFrame, df_pred_ctrl: pd.DataFrame,
                                    models: Dict[str, object], cfg: Config) -> pd.DataFrame:
    """
    For each drug model, compute how ΔPC aligns with model coefficients (PC part only):
      effect_score = ΔPC ⋅ beta_PCs   (positive means shift towards higher LN_IC50 if beta positive)
    Then correlate baseline predicted LN_IC50 (from controls) vs effect_score across lines.
    Returns one row per (drug_model, treated_drug) with correlations.
    """
    pcs = list_pc_columns(cfg)
    # Collect beta_PCs for each model (ignoring intercept)
    beta_pcs = {}
    for d, model in models.items():
        # model.coef_ is aligned with feats list order; we need to pick PC entries
        # We'll rebuild mapping by refitting a dummy vector of ones and checking columns,
        # but simpler: assume feats order (PCs first, then tissues, then growth)
        # This must match feature_columns() construction.
        # Extract the first n_pcs coefficients:
        beta_pcs[d] = np.array(model.coef_[:cfg.n_pcs])

    # For each treated drug, compute effect scores per line for each model
    rows = []
    for (line, tdrug), sub in df_delta.groupby([cfg.sanger_col, "treated_drug"]):
        delta_vec = sub[[f"DELTA_{pc}" for pc in pcs]].values.squeeze()
        # Baseline prediction(s) for this line from controls: average across its control cells
        base = (df_pred_ctrl[df_pred_ctrl[cfg.sanger_col] == line]
                .groupby("drug_model")["pred_LN_IC50"].mean())
        for d in beta_pcs.keys():
            if d not in base.index:
                continue
            effect = float(delta_vec @ beta_pcs[d])
            rows.append({
                cfg.sanger_col: line,
                "treated_drug": tdrug,
                "drug_model": d,
                "effect_score": effect,
                "baseline_pred_LN_IC50": float(base.loc[d]),
            })
    df_effects = pd.DataFrame(rows)
    if df_effects.empty:
        return df_effects

    # Correlations per (drug_model, treated_drug)
    results = []
    for (d, tdrug), grp in df_effects.groupby(["drug_model", "treated_drug"]):
        if len(grp) < 3:
            continue
        x = grp["baseline_pred_LN_IC50"].values
        y = grp["effect_score"].values
        r = np.corrcoef(x, y)[0, 1]
        # Spearman
        from scipy.stats import spearmanr
        rs, p_s = spearmanr(x, y)
        results.append({
            "drug_model": d,
            "treated_drug": tdrug,
            "n_lines": len(grp),
            "pearson_r": r,
            "spearman_r": float(rs),
            "spearman_p": float(p_s)
        })
    return pd.DataFrame(results)


# -----------------------------
# Optional: back-project coefficients to gene space
# -----------------------------

def backproject_beta_to_genes(models: Dict[str, object], cfg: Config) -> Optional[pd.DataFrame]:
    """
    If PCA loadings for the *bulk* PCA are provided (genes x PCs), back-project model PC-coefficients to gene space:
      beta_genes = loadings (genes x PCs) @ beta_PCs (PCs,)
    Returns long-form table with ['gene_id','drug_model','beta_gene'].
    """
    if not cfg.bulk_pca_loadings_path:
        return None

    loadings = read_table(cfg.bulk_pca_loadings_path)
    # Expect columns PC1..PCn and a gene ID column
    pcs = list_pc_columns(cfg)
    assert all(pc in loadings.columns for pc in pcs), "PCA loadings missing some PC columns."

    rows = []
    for d, model in models.items():
        beta_pcs = np.array(model.coef_[:cfg.n_pcs]).reshape(-1, 1)  # (p,1)
        beta_genes = loadings[pcs].values @ beta_pcs  # (n_genes,1)
        tmp = pd.DataFrame({
            cfg.gene_id_col: loadings[cfg.gene_id_col].values,
            "drug_model": d,
            "beta_gene": beta_genes.squeeze()
        })
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


# -----------------------------
# Plotting helpers (matplotlib only; one plot per figure; no styles/colors set)
# -----------------------------

def plot_pred_distribution(df_pred_ctrl: pd.DataFrame, outdir: str):
    plt.figure(figsize=(5,4))
    df_pred_ctrl.boxplot(column="pred_LN_IC50", by="drug_model", rot=90)
    plt.suptitle("")
    plt.title("Tahoe controls: predicted LN_IC50 by drug model")
    plt.xlabel("Drug model")
    plt.ylabel("Predicted LN_IC50")
    savefig(os.path.join(outdir, "pred_distributions_by_drug.png"))


def plot_scatter_baseline_vs_effect(df_corr_src: pd.DataFrame, df_effects_src: pd.DataFrame, outdir: str):
    # Make one scatter per (drug_model, treated_drug); only a few to avoid explosion; choose top by |spearman_r|
    if df_corr_src.empty:
        return
    top = (df_corr_src.copy()
           .assign(abs_r=lambda d: d["spearman_r"].abs())
           .sort_values("abs_r", ascending=False).head(6))

    for _, row in top.iterrows():
        d = row["drug_model"]; tdrug = row["treated_drug"]
        sub = df_effects_src[(df_effects_src["drug_model"] == d) & (df_effects_src["treated_drug"] == tdrug)]
        plt.figure(figsize=(4.5,4))
        plt.scatter(sub["baseline_pred_LN_IC50"], sub["effect_score"])
        plt.xlabel("Baseline predicted LN_IC50 (controls)")
        plt.ylabel("Δ-effect score (ΔPC ⋅ β)")
        plt.title(f"{d} model vs. {tdrug} treatment")
        savefig(os.path.join(outdir, f"scatter_{d}_vs_{tdrug}.png"))


# -----------------------------
# Main
# -----------------------------

def main(cfg: Config):
    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.artifacts_dir)

    # 1) Load
    print("[io] Loading tables...")
    df_tahoe = read_table(cfg.tahoe_parquet)
    df_bulk = read_table(cfg.bulk_table_path)

    # Optional: restrict to selected drugs
    if cfg.restrict_to_drugs is not None:
        df_bulk = df_bulk[df_bulk[cfg.bulk_drug_col].isin(cfg.restrict_to_drugs)].copy()

    # 2) (Optional) Align Tahoe PCs to bulk PCs with Procrustes using overlaps
    if cfg.do_procrustes_alignment:
        print("[align] Aligning Tahoe PCs to bulk PCs (orthogonal Procrustes)...")
        df_tahoe, align_params = align_tahoe_to_bulk(df_tahoe, df_bulk, cfg)
        with open(os.path.join(cfg.output_dir, "alignment_params.json"), "w") as f:
            json.dump(align_params, f, indent=2)

    # 3) Standardize features
    feats_bulk = feature_columns(cfg, df_bulk)
    feats_tahoe = feature_columns(cfg, df_tahoe)
    missing_in_tahoe = [c for c in feats_bulk if c not in df_tahoe.columns]
    if missing_in_tahoe:
        raise ValueError(f"Tahoe missing feature columns needed by bulk: {missing_in_tahoe}")

    print("[scale] Fitting scaler on bulk features...")
    scaler_path = os.path.join(cfg.artifacts_dir, "scaler.joblib")
    if os.path.exists(scaler_path) and not cfg.train_models_if_missing:
        scaler = load(scaler_path)
    else:
        scaler = fit_scaler_on_bulk(df_bulk, cfg)
        dump(scaler, scaler_path)

    print("[scale] Standardizing bulk and Tahoe using bulk scaler...")
    df_bulk_std = df_bulk.copy()
    df_tahoe_std = df_tahoe.copy()
    standardize_df_inplace(df_bulk_std, scaler, cfg)
    standardize_df_inplace(df_tahoe_std, scaler, cfg)

    # 4) Train or load models
    if cfg.train_models_if_missing:
        print("[train] Training per-drug ElasticNet models on bulk...")
        model_paths = train_or_load_models(df_bulk_std, cfg)
    else:
        model_paths = {}  # may be unused if you load below

    # Load all models present in artifacts_dir
    models, feats = load_models_from_dir(cfg.artifacts_dir)

    # 5) Predict on Tahoe controls
    print("[infer] Predicting on Tahoe controls...")
    df_pred_ctrl = predict_on_tahoe_controls(df_tahoe_std, models, cfg)
    df_pred_ctrl.to_csv(os.path.join(cfg.output_dir, "tahoe_controls_predictions.csv"), index=False)

    # QC plot: boxplot per drug model
    plot_pred_distribution(df_pred_ctrl, cfg.output_dir)

    # 6) ΔPC for treated vs control
    print("[perturb] Computing ΔPC (treated - control) by line & drug...")
    df_delta = delta_pc_by_line_and_drug(df_tahoe_std, cfg)
    df_delta.to_csv(os.path.join(cfg.output_dir, "delta_pc_by_line_and_drug.csv"), index=False)

    # 7) Project ΔPC onto model β (PC part) and correlate with baseline preds
    print("[perturb] Projecting ΔPC onto β and correlating...")
    df_effects = []
    # Build once and reuse
    effects_rows = []
    pcs = list_pc_columns(cfg)
    beta_pcs_map = {d: np.array(m.coef_[:cfg.n_pcs]) for d, m in models.items()}
    # Prepare baseline per line per drug model
    base = (df_pred_ctrl.groupby([cfg.sanger_col, "drug_model"])["pred_LN_IC50"]
            .mean().reset_index())
    # Iterate df_delta rows
    for _, r in df_delta.iterrows():
        line = r[cfg.sanger_col]
        tdrug = r["treated_drug"]
        delta_vec = r[[f"DELTA_{pc}" for pc in pcs]].values
        for d, beta_vec in beta_pcs_map.items():
            bline = base[(base[cfg.sanger_col] == line) & (base["drug_model"] == d)]
            if bline.empty:
                continue
            effect = float(delta_vec @ beta_vec)
            effects_rows.append({
                cfg.sanger_col: line,
                "treated_drug": tdrug,
                "drug_model": d,
                "effect_score": effect,
                "baseline_pred_LN_IC50": float(bline["pred_LN_IC50"].values[0])
            })
    df_effects = pd.DataFrame(effects_rows)
    df_effects.to_csv(os.path.join(cfg.output_dir, "effects_baseline_vs_delta.csv"), index=False)

    # Correlations
    from scipy.stats import spearmanr
    corr_rows = []
    for (d, tdrug), grp in df_effects.groupby(["drug_model", "treated_drug"]):
        if len(grp) >= 3:
            r = np.corrcoef(grp["baseline_pred_LN_IC50"], grp["effect_score"])[0, 1]
            rs, p_s = spearmanr(grp["baseline_pred_LN_IC50"], grp["effect_score"])
            corr_rows.append({
                "drug_model": d,
                "treated_drug": tdrug,
                "n_lines": len(grp),
                "pearson_r": float(r),
                "spearman_r": float(rs),
                "spearman_p": float(p_s)
            })
    df_corr = pd.DataFrame(corr_rows).sort_values(["drug_model", "treated_drug"])
    df_corr.to_csv(os.path.join(cfg.output_dir, "baseline_vs_delta_correlations.csv"), index=False)

    # Scatter plots for top associations
    plot_scatter_baseline_vs_effect(df_corr, df_effects, cfg.output_dir)

    # 8) Optional: back-project to gene space (if loadings provided)
    if cfg.bulk_pca_loadings_path:
        print("[genes] Back-projecting β to genes...")
        df_beta_genes = backproject_beta_to_genes(models, cfg)
        if df_beta_genes is not None:
            df_beta_genes.to_csv(os.path.join(cfg.output_dir, "beta_genes_long.csv"), index=False)

    print("[done] Outputs written to:", cfg.output_dir)


if __name__ == "__main__":
    # Load config JSON if provided next to the script
    cfg_path = os.environ.get("TRANSFER_PIPELINE_CONFIG", "config_transfer_pipeline.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg_dict = json.load(f)
        cfg = Config(**cfg_dict)
    else:
        cfg = Config()
    main(cfg)
