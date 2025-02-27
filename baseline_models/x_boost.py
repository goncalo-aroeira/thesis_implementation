import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# ====== Step 1: Load Data ====== #
# Load gene embeddings (pretrained scGPT embeddings)
gene_embeddings = pd.read_csv("/home/goncalo/scgpt/scGPT/whole_human_model/gene_embeddings.csv", index_col=0)

# Load pseudo-bulk gene expression data (expression per cell line)
pseudo_bulk_expression = pd.read_csv("/home/goncalo/scgpt/datasets/pseudo_bulk/pseudo_bulk_filtered.csv", index_col=0)

# Load IC50 values from GDSC
gdsc_ic50 = pd.read_csv("/home/goncalo/scgpt/datasets/gdsc/gdsc_final_cleaned.csv")

# ====== Step 2: Align Genes ====== #
# Find common genes in both datasets
common_genes = list(set(pseudo_bulk_expression.columns) & set(gene_embeddings.index))

# Select only matching genes
filtered_expression = pseudo_bulk_expression[common_genes]  # (num_cell_lines, num_common_genes)
filtered_embeddings = gene_embeddings.loc[common_genes]  # (num_common_genes, embedding_dim)

# ====== Step 3: Transform Expression Using Gene Embeddings ====== #
# Weighted sum of gene embeddings based on expression levels
pseudo_bulk_embeddings = np.dot(filtered_expression, filtered_embeddings.values) / filtered_expression.sum(axis=1).values[:, None]

# Convert to DataFrame
pseudo_bulk_embeddings_df = pd.DataFrame(pseudo_bulk_embeddings, index=pseudo_bulk_expression.index)

# Save transformed dataset
pseudo_bulk_embeddings_df.to_csv("pseudo_bulk_gene_embeddings.csv")
print(f"✅ Transformed gene embeddings saved! Shape: {pseudo_bulk_embeddings_df.shape}")

# ====== Step 4: Align with IC50 Data ====== #
# Find common cell lines between transformed embeddings and IC50 dataset
common_cells = list(set(pseudo_bulk_embeddings_df.index) & set(gdsc_ic50["SANGER_MODEL_ID"]))

# Select matching data
X = pseudo_bulk_embeddings_df.loc[common_cells].values  # Transformed gene embeddings as input
y = gdsc_ic50[gdsc_ic50["SANGER_MODEL_ID"].isin(common_cells)]["LN_IC50"].values  # IC50 target

# ====== Step 5: Train-Test Split ====== #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ====== Step 6: Train XGBoost Model ====== #
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# ====== Step 7: Evaluate Model ====== #
y_pred = xgb_model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"✅ Final Model RMSE: {rmse:.4f}")
