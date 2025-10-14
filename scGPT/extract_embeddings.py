import pandas as pd
import numpy as np
import scanpy as sc
import torch
from pathlib import Path
from scgpt.tasks.cell_emb import embed_data

# === CONFIG ===
DATA_TYPE = "bulk"  # or "single-cell"
PARQUET_PATH = "data/cell_gene_matrix_tpm.parquet"
MAPPING_PATH = "data/gene_id_to_symbol_mapping.csv"
MODEL_DIR = Path("examples/save/")
GENE_COL_NAME = "feature_name"
OUTPUT_H5AD = "embedded_data.h5ad"

print("🚀 Starting embedding pipeline...")

# === Load expression data ===
print(f"📂 Reading expression matrix: {PARQUET_PATH}")
df = pd.read_parquet(PARQUET_PATH)

if "gene_id" in df.columns:
    print("🔎 Found 'gene_id' column — setting as index and transposing...")
    df = df.set_index("gene_id").T

print(f"✅ Loaded data shape: {df.shape} (samples x genes)")

# === Load and apply gene mapping ===
print(f"📘 Loading gene ID to symbol mapping from: {MAPPING_PATH}")
gene_map = pd.read_csv(MAPPING_PATH)
gene_dict = dict(zip(gene_map["gene_id"], gene_map["symbol"]))

print("🔁 Mapping internal gene IDs to symbols...")
df.rename(columns=gene_dict, inplace=True)

# Drop any columns with missing mapping or duplicate gene symbols
original_cols = df.shape[1]
df = df.loc[:, ~df.columns.duplicated()]  # remove duplicates
print(f"✅ Renamed and deduplicated gene columns: {df.shape[1]} (was {original_cols})")

# === Create AnnData ===
print("📐 Creating AnnData object...")
adata = sc.AnnData(X=df.values.astype(np.float32))
adata.obs_names = df.index.astype(str)
adata.var_names = df.columns.astype(str)
adata.var[GENE_COL_NAME] = adata.var_names
print(f"✅ AnnData created with shape: {adata.shape}")

# === Preprocessing ===
print(f"🧪 Preprocessing data for type: {DATA_TYPE}")
if DATA_TYPE == "bulk":
    adata.X = np.log2(adata.X + 1)
    print("🔧 Applied log2(TPM + 1) for bulk input.")
elif DATA_TYPE == "single-cell":
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print("🔧 Normalized and log1p-transformed single-cell input.")
else:
    raise ValueError("❌ DATA_TYPE must be 'bulk' or 'single-cell'")

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# === Run embedding ===
print("🧠 Running scGPT embedding extraction...")
embedded_adata = embed_data(
    adata_or_file=adata,
    model_dir=MODEL_DIR,
    gene_col=GENE_COL_NAME,
    max_length=40000,
    batch_size=32,
    device=device,
    return_new_adata=True
)

print("✅ Embedding extraction complete.")
print(f"🧬 Embedded data shape: {embedded_adata.shape}")

# === Save ===
embedded_adata.write(OUTPUT_H5AD)
print(f"💾 Saved embedded AnnData to: {OUTPUT_H5AD}")
'''

from scgpt.tasks.cell_emb import embed_data
import scanpy as sc
from pathlib import Path
import torch

# Load your preprocessed AnnData
adata = sc.read("data/pancancer_dimred.h5ad")
print(f"✅ Loaded final processed data: {adata.shape}")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Run scGPT embedding
embedded_adata = embed_data(
    adata_or_file=adata,
    model_dir=Path("examples/save/"),
    gene_col="index",  # if var_names are gene symbols
    max_length=30000,
    device=device,
    return_new_adata=True
)

# Save embedded output
embedded_adata.write("pancancer_embeddings.h5ad")
print("✅ Saved scGPT embeddings to pancancer_embeddings.h5ad")
'''