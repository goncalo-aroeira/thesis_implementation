import pandas as pd

# Load GDSC bulk RNA-seq data
gdsc_bulk = pd.read_csv("gdsc/gdsc_final_cleaned.csv")

# Load single-cell RNA-seq data
sc_data = pd.read_csv("sc_data/rnaseq_all_data.csv")

# Check the first few rows
print(gdsc_bulk.head())
print(sc_data.head())

# Check common cell line IDs (SANGER_MODEL_ID / model_id)
common_cell_lines = set(gdsc_bulk["SANGER_MODEL_ID"]) & set(sc_data["model_id"])
print(f"Common Cell Lines: {len(common_cell_lines)}")

# Check unique genes in single-cell data
print(f"Unique genes in single-cell data: {sc_data['gene_symbol'].nunique()}")

# Filter single-cell data to only include model_ids that match GDSC SANGER_MODEL_ID
sc_filtered = sc_data[sc_data["model_id"].isin(gdsc_bulk["SANGER_MODEL_ID"])]

print(f"Filtered single-cell dataset shape: {sc_filtered.shape}")

# Aggregate by model_id (matching SANGER_MODEL_ID) and gene_symbol
pseudo_bulk = sc_filtered.groupby(["model_id", "gene_symbol"]).agg({
    "fpkm": "mean"  # Use the average expression
}).reset_index()

print(f"Pseudo-bulk single-cell data shape: {pseudo_bulk.shape}")

# Find common genes between the two datasets
common_genes = set(pseudo_bulk["gene_symbol"]) & set(gdsc_bulk.columns)
print(f"Number of common genes: {len(common_genes)}")

# Filter pseudo-bulk dataset to only include common genes
pseudo_bulk_filtered = pseudo_bulk[pseudo_bulk["gene_symbol"].isin(common_genes)]

# Pivot table so that rows = cell lines, columns = genes
pseudo_bulk_pivot = pseudo_bulk_filtered.pivot(index="model_id", columns="gene_symbol", values="fpkm").fillna(0)

print(f"Final pseudo-bulk shape (cells × genes): {pseudo_bulk_pivot.shape}")
