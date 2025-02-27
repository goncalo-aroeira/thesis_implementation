import pandas as pd
"""
# Load the cell-gene matrix
file_path = "rnaseq_fpkm.csv"  # Replace with your actual file path
cell_gene_matrix = pd.read_csv(file_path)

# Ensure the first column is correctly named
if cell_gene_matrix.columns[0] != "model_id":
    cell_gene_matrix.rename(columns={cell_gene_matrix.columns[0]: "model_id"}, inplace=True)

# Check for duplicate model_id (cell lines)
duplicate_models = cell_gene_matrix.duplicated(subset=["model_id"], keep=False)

# Get and print the duplicated entries
duplicated_entries = cell_gene_matrix[duplicate_models]

if not duplicated_entries.empty:
    print(f"🔍 Found {duplicated_entries.shape[0]} duplicated rows for model_id.")
    print(duplicated_entries.head())  # Display a sample of duplicates
else:
    print("✅ No duplicate model_ids found - likely a pseudo-bulk dataset.")
"""


# Load the dataset
file_path = "rnaseq_all_data.csv"  # Replace with the actual file path
df = pd.read_csv(file_path)

# Check if the same gene appears multiple times per model_id
duplicate_counts = df.groupby(["model_id", "gene_id"]).size().reset_index(name="count")

# If we find duplicate (model_id, gene_id) pairs, it suggests single-cell resolution
duplicates = duplicate_counts[duplicate_counts["count"] > 1]

if not duplicates.empty:
    print("✅ The dataset is likely SINGLE-CELL, as genes are measured multiple times per model_id.")
else:
    print("❌ The dataset is likely PSEUDO-BULK, as each gene appears only once per model_id.")

# Check if individual `model_id`s appear multiple times in different rows (should be true for single-cell)
model_counts = df["model_id"].value_counts()

if model_counts.max() > 1:
    print("✅ The dataset contains multiple measurements per model_id, supporting single-cell resolution.")
else:
    print("❌ The dataset does not contain multiple entries per model_id, suggesting pseudo-bulk.")
