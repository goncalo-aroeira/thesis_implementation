import pandas as pd
import numpy as np

# ===========================
# 1️⃣ Load and Optimize Datasets
# ===========================

# Load GDSC bulk data (drug response)
gdsc_bulk = pd.read_csv(
    "gdsc/gdsc_final_cleaned.csv",
    usecols=["SANGER_MODEL_ID", "DRUG_ID", "LN_IC50"],
    dtype={"SANGER_MODEL_ID": "str", "DRUG_ID": "int32", "LN_IC50": "float32"}
)

# Load Single-Cell RNA-seq Data
sc_data = pd.read_csv(
    "sc_data/rnaseq_all_data.csv",
    usecols=["model_id", "gene_symbol", "fpkm"],
    dtype={"model_id": "str", "gene_symbol": "str", "fpkm": "float32"}
)

print(f"GDSC Bulk Shape: {gdsc_bulk.shape}")
print(f"Single-Cell Data Shape: {sc_data.shape}")

# ===========================
# 2️⃣ Aggregate Single-Cell Data into Pseudo-Bulk
# ===========================

# Filter single-cell data to include only cell lines present in GDSC
sc_filtered = sc_data[sc_data["model_id"].isin(gdsc_bulk["SANGER_MODEL_ID"])]

# Aggregate gene expression per cell line
pseudo_bulk = sc_filtered.groupby(["model_id", "gene_symbol"])["fpkm"].mean().reset_index()

# Pivot table so that rows = cell lines, columns = genes
pseudo_bulk_pivot = pseudo_bulk.pivot(index="model_id", columns="gene_symbol", values="fpkm").fillna(0)

# Rename index to match GDSC format
pseudo_bulk_pivot.index.name = "SANGER_MODEL_ID"

# Convert to DataFrame
pseudo_bulk_pivot.reset_index(inplace=True)

print(f"Pseudo-Bulk Shape Before Filtering: {pseudo_bulk_pivot.shape}")

# ===========================
# 3️⃣ Select Top 2,000 Highly Variable Genes (HVGs)
# ===========================

# Compute variance only for gene expression columns
gene_variances = pseudo_bulk_pivot.drop(columns=["SANGER_MODEL_ID"]).var(axis=0)

# Select the top 2,000 most variable genes
top_variable_genes = gene_variances.nlargest(2000).index

# Keep only these genes
pseudo_bulk_pivot = pseudo_bulk_pivot[["SANGER_MODEL_ID"] + list(top_variable_genes)]

print(f"Reduced Pseudo-Bulk Shape: {pseudo_bulk_pivot.shape}")


# ===========================
# 4️⃣ Merge GDSC Drug Response Data with Pseudo-Bulk
# ===========================

# Save pseudo-bulk to a CSV file
pseudo_bulk_pivot.to_csv("pseudo_bulk/pseudo_bulk_filtered.csv", index=False)
print("Saved pseudo-bulk dataset to disk.")

# Free memory by reloading from disk
del pseudo_bulk_pivot

# Reload from disk
pseudo_bulk_pivot = pd.read_csv("pseudo_bulk/pseudo_bulk_filtered.csv")

print("Pseudo-bulk dataset reloaded successfully!")
print(f"Pseudo-Bulk Shape: {pseudo_bulk_pivot.shape}")

chunk_size = 100000  # Adjust based on available RAM

# Process GDSC bulk data in chunks
for i, chunk in enumerate(pd.read_csv("gdsc/gdsc_final_cleaned.csv", chunksize=100000, dtype={"LN_IC50": "float32"}, low_memory=False)):
    chunk = chunk.merge(pseudo_bulk_pivot, on="SANGER_MODEL_ID", how="left")
    chunk.to_csv(f"pseudo_bulk/aligned_chunk_{i}.csv", index=False)
    print(f"Processed and saved chunk {i+1}")

import pandas as pd
import glob

# Get all chunk files
chunk_files = sorted(glob.glob("pseudo_bulk/aligned_chunk_*.csv"))

# Open a new CSV file and write chunks directly
with open("pseudo_bulk/gdsc_single_cell_aligned.csv", "w", newline="") as outfile:
    writer = None

    for i, file in enumerate(chunk_files):
        print(f"Merging chunk {i+1}/{len(chunk_files)}")

        # Read only one chunk at a time
        with open(file, "r") as chunk:
            if i == 0:
                # Write header for the first chunk
                outfile.write(chunk.read())
            else:
                # Skip the header for other chunks
                next(chunk)  # Skip first line (header)
                outfile.write(chunk.read())

print("Final aligned dataset saved successfully!")
