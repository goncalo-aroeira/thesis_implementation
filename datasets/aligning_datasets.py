import pandas as pd
import numpy as np
import glob

# ===========================
# 1️⃣ Load & Optimize Datasets
# ===========================

# Load GDSC bulk drug response data
gdsc_bulk = pd.read_csv(
    "gdsc/gdsc_final_cleaned.csv",
    usecols=["SANGER_MODEL_ID", "DRUG_ID", "LN_IC50"],
    dtype={"SANGER_MODEL_ID": "str", "DRUG_ID": "int32", "LN_IC50": "float32"}
)

# Load Precomputed Cell by Gene Matrix (TPM values)
cell_gene_matrix = pd.read_csv(
    "sc_data/rnaseq_fpkm.csv",  # Change to actual filename
    index_col=0  # Assuming model_id is the first column
)

# Ensure model_id column matches GDSC format
cell_gene_matrix.index.name = "SANGER_MODEL_ID"

print(f"✅ GDSC Bulk Shape: {gdsc_bulk.shape}")
print(f"✅ Cell by Gene Matrix Shape: {cell_gene_matrix.shape}")

# ===========================
# 2️⃣ Normalize Gene Expression
# ===========================

# Option 1: Normalize Before Aggregation (Log-T Normalize)
normalized_before = np.log1p(cell_gene_matrix)

# Option 2: Aggregate First, Then Normalize (Pseudo-Bulk Mean)
pseudo_bulk = cell_gene_matrix.groupby(cell_gene_matrix.index).mean()

# Normalize After Aggregation
normalized_after = np.log1p(pseudo_bulk)

# Save both versions for evaluation
normalized_before.to_csv("pseudo_bulk/pseudo_bulk_normalized_before.csv")
normalized_after.to_csv("pseudo_bulk/pseudo_bulk_normalized_after.csv")

print("🔹 Normalization completed. Two versions saved: Before and After Aggregation.")

# ===========================
# 3️⃣ Select Top 2,000 Highly Variable Genes (HVGs)
# ===========================

# Compute variance for both normalization methods
var_before = normalized_before.var(axis=0).nlargest(2000).index
var_after = normalized_after.var(axis=0).nlargest(2000).index

# Keep only highly variable genes
filtered_before = normalized_before[var_before].reset_index()
filtered_after = normalized_after[var_after].reset_index()

print(f"✅ Filtered Pseudo-Bulk Shape (Before Normalization): {filtered_before.shape}")
print(f"✅ Filtered Pseudo-Bulk Shape (After Normalization): {filtered_after.shape}")

# Save both filtered datasets
filtered_before.to_csv("pseudo_bulk/pseudo_bulk_filtered_before.csv", index=False)
filtered_after.to_csv("pseudo_bulk/pseudo_bulk_filtered_after.csv", index=False)

"""
# ===========================
# 4️⃣ Merge GDSC Drug Response Data with Pseudo-Bulk
# ===========================

# Choose the preferred dataset (before or after normalization)
pseudo_bulk_final = filtered_after  # Change to `filtered_before` if needed

# Merge with GDSC drug response data
merged_data = gdsc_bulk.merge(pseudo_bulk_final, on="SANGER_MODEL_ID", how="left")

# Save final dataset
merged_data.to_csv("pseudo_bulk/gdsc_single_cell_aligned.csv", index=False)

print("✅ Final aligned dataset saved successfully!")
print(f"📌 Final Shape: {merged_data.shape}")
"""