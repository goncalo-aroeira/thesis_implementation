import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# ====== Step 1: Load Processed Gene Embeddings (Cell Line Representations) ====== #
pseudo_bulk_embeddings_df = pd.read_csv("pseudo_bulk_gene_embeddings.csv", index_col=0)

# ====== Step 2: Load IC50 Data (GDSC) ====== #
gdsc_ic50 = pd.read_csv("/home/goncalo/scgpt/datasets/gdsc/gdsc_final_cleaned.csv")

# ✅ Keep all drug-cell line pairs (Do NOT aggregate IC50 values)
# Ensure cell-line embeddings exist for each IC50 entry
common_cells = list(set(pseudo_bulk_embeddings_df.index) & set(gdsc_ic50["SANGER_MODEL_ID"]))

# Filter IC50 data to keep only cell lines present in embeddings
gdsc_ic50_filtered = gdsc_ic50[gdsc_ic50["SANGER_MODEL_ID"].isin(common_cells)]

# Expand gene embeddings to match (cell line, drug) pairs
X = np.array([pseudo_bulk_embeddings_df.loc[cell_line].values for cell_line in gdsc_ic50_filtered["SANGER_MODEL_ID"]])

# IC50 values as targets
y = gdsc_ic50_filtered["LN_IC50"].values

print(f"✅ Matched Data: X shape {X.shape}, y shape {y.shape}")

# ====== Step 3: Train-Test Split ====== #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ====== Step 4: Train XGBoost Model ====== #
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# Save model after training
xgb_model.save_model("xgboost_trained_model.json")
print("✅ Model saved successfully!")

# ====== Step 5: Evaluate Model ====== #
y_pred = xgb_model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"✅ Final Model RMSE: {rmse:.4f}")
