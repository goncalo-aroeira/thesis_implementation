import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


# Load datasets
sc_data = pd.read_csv("sc_data/rnaseq_all_data.csv", nrows=5)
pseudo_bulk = pd.read_csv("pseudo_bulk/gdsc_single_cell_aligned.csv", nrows=5)

# Print shapes of full datasets
print("\n📊 Dataset Shapes:")

# Single-Cell RNA-seq Data
sc_shape = pd.read_csv("sc_data/rnaseq_all_data.csv").shape
print(f"🔹 Single-Cell RNA-seq Data: {sc_shape[0]:,} rows, {sc_shape[1]} columns")

# Pseudo-Bulk Data
pb_shape = pd.read_csv("pseudo_bulk/gdsc_single_cell_aligned.csv").shape
print(f"🔹 Pseudo-Bulk Data: {pb_shape[0]:,} rows, {pb_shape[1]} columns")

# GDSC Bulk Data
gdsc_shape = pd.read_csv("gdsc/gdsc_final_cleaned.csv").shape
print(f"🔹 GDSC Bulk Data: {gdsc_shape[0]:,} rows, {gdsc_shape[1]} columns")

print(f"🔹 Single-Cell Data Columns: {len(sc_data.columns)}")
print(f"🔹 Pseudo-Bulk Data Columns: {len(pseudo_bulk.columns)}")

print("🔹 Single-Cell Data Types:")
print(sc_data.dtypes)

print("🔹 Pseudo-Bulk Data Types:")
print(pseudo_bulk.dtypes)


"""
# ===========================
# 1️⃣ Create Output Directory
# ===========================

# Create a directory to store statistics and plots
output_dir = "statistics"
os.makedirs(output_dir, exist_ok=True)

# ===========================
# 2️⃣ Load Datasets
# ===========================

# Load GDSC bulk dataset (drug response)
gdsc_bulk = pd.read_csv("gdsc/gdsc_final_cleaned.csv", usecols=["SANGER_MODEL_ID", "DRUG_ID", "LN_IC50"])
# Load Single-Cell RNA-seq Data
sc_data = pd.read_csv("sc_data/rnaseq_all_data.csv", usecols=["model_id", "gene_symbol", "fpkm"])
# Load Final Merged Dataset
final_data = pd.read_csv("pseudo_bulk/gdsc_single_cell_aligned.csv")

# ===========================
# 3️⃣ Compute Dataset Statistics
# ===========================

# GDSC Bulk Statistics
num_gdsc_cell_lines = gdsc_bulk["SANGER_MODEL_ID"].nunique()
num_gdsc_drugs = gdsc_bulk["DRUG_ID"].nunique()
num_gdsc_pairs = gdsc_bulk.shape[0]  # Each row is a (cell line, drug) pair

# Single-Cell Statistics
num_sc_cells = sc_data["model_id"].nunique()
num_sc_genes = sc_data["gene_symbol"].nunique()

# Final Merged Dataset Statistics
num_final_cell_lines = final_data["SANGER_MODEL_ID"].nunique()
num_final_drugs = final_data["DRUG_ID"].nunique()
num_final_pairs = final_data.shape[0]
num_final_genes = final_data.shape[1] - 3  # Excluding non-gene columns

# Save dataset statistics to CSV
stats_dict = {
    "Total Cell Lines (GDSC)": num_gdsc_cell_lines,
    "Total Unique Drugs (GDSC)": num_gdsc_drugs,
    "Total (Cell Line, Drug) Pairs": num_gdsc_pairs,
    "Total Unique Genes in Single-Cell": num_sc_genes,
    "Total Unique Cell Lines in Single-Cell": num_sc_cells,
    "Total Genes After HVG Selection": num_final_genes,
    "Total Cell Lines in Final Dataset": num_final_cell_lines,
    "Total Unique Drugs in Final Dataset": num_final_drugs,
    "Total (Cell Line, Drug) Pairs in Final Dataset": num_final_pairs
}

stats_df = pd.DataFrame(stats_dict.items(), columns=["Metric", "Value"])
stats_df.to_csv(f"{output_dir}/dataset_statistics.csv", index=False)

print("📂 Dataset statistics saved to 'statistics/dataset_statistics.csv' 🎉")

# ===========================
# 4️⃣ Generate & Save Plots
# ===========================

# Plot IC50 Distribution
plt.figure(figsize=(8, 5))
sns.histplot(final_data["LN_IC50"], bins=50, kde=True, color="blue")
plt.xlabel("Log IC50")
plt.ylabel("Frequency")
plt.title("Distribution of Log IC50 Values")
plt.grid()
plt.savefig(f"{output_dir}/ic50_distribution.png")
plt.close()

# Compute Correlation Between Gene Expression and IC50
gene_columns = final_data.columns[3:]  # Exclude metadata columns
correlations = final_data[gene_columns].corrwith(final_data["LN_IC50"]).sort_values()

# Save top 10 correlated genes
correlations_df = pd.DataFrame({
    "Top Positively Correlated Genes": correlations.tail(10).index.tolist(),
    "Top Negatively Correlated Genes": correlations.head(10).index.tolist()
})
correlations_df.to_csv(f"{output_dir}/gene_ic50_correlations.csv", index=False)

# Plot Drug Testing Frequency
drug_counts = final_data.groupby("DRUG_ID")["SANGER_MODEL_ID"].nunique().sort_values(ascending=False)
plt.figure(figsize=(10, 5))
sns.histplot(drug_counts, bins=50, kde=True, color="green")
plt.xlabel("Number of Cell Lines per Drug")
plt.ylabel("Frequency")
plt.title("Distribution of Cell Line Testing Per Drug")
plt.grid()
plt.savefig(f"{output_dir}/cell_lines_per_drug.png")
plt.close()

# Compute Gene Expression Variability
gene_std = final_data.iloc[:, 3:].std().sort_values(ascending=False)

# Save most variable genes
gene_std_df = pd.DataFrame({"Gene": gene_std.index, "Standard Deviation": gene_std.values})
gene_std_df.to_csv(f"{output_dir}/most_variable_genes.csv", index=False)

# PCA Visualization of Cell Line Clustering
pca = PCA(n_components=2)
pca_results = pca.fit_transform(final_data.iloc[:, 3:])
plt.figure(figsize=(8, 6))
plt.scatter(pca_results[:, 0], pca_results[:, 1], alpha=0.5, color="purple")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA Clustering of Cell Lines by Gene Expression")
plt.grid()
plt.savefig(f"{output_dir}/pca_cell_line_clustering.png")
plt.close()

# ===========================
# 5️⃣ Final Summary & Next Steps
# ===========================

print("\n🎯 Dataset analysis completed! The following files have been saved:")
print(f"- 📊 Dataset statistics: {output_dir}/dataset_statistics.csv")
print(f"- 🔬 Gene-IC50 correlations: {output_dir}/gene_ic50_correlations.csv")
print(f"- 🧬 Most variable genes: {output_dir}/most_variable_genes.csv")
print(f"- 📈 IC50 Distribution Plot: {output_dir}/ic50_distribution.png")
print(f"- 🔥 Drug Testing Frequency Plot: {output_dir}/cell_lines_per_drug.png")
print(f"- 🎨 PCA Clustering Plot: {output_dir}/pca_cell_line_clustering.png")
"""