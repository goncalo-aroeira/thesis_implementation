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

from scipy.stats import pearsonr
correlation, _ = pearsonr(y_test, y_pred)
print(f"✅ Pearson Correlation: {correlation:.4f}")

from scipy.stats import spearmanr
spearman_corr, _ = spearmanr(y_test, y_pred)
print(f"✅ Spearman Correlation: {spearman_corr:.4f}")

import matplotlib.pyplot as plt
xgb.plot_importance(xgb_model, max_num_features=20)  # Show top 20 most important genes
plt.show()

import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("True IC50")
plt.ylabel("Predicted IC50")
plt.title("True vs. Predicted IC50 Values")
plt.show()


#Radom Forest
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Compute RMSE
rf_rmse = mean_squared_error(y_test, rf_pred, squared=False)
print(f"✅ Random Forest RMSE: {rf_rmse:.4f}")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

mlp_model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(1)  # Output layer for IC50 prediction
])

mlp_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
mlp_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate MLP Model
mlp_pred = mlp_model.predict(X_test).flatten()
mlp_rmse = mean_squared_error(y_test, mlp_pred, squared=False)
print(f"✅ MLP RMSE: {mlp_rmse:.4f}")


from sklearn.decomposition import PCA

pca = PCA(n_components=100)  # Reduce to 100 principal components
X_pca = pca.fit_transform(X)

# Train XGBoost on PCA-transformed features
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)
xgb_model.fit(X_train_pca, y_train_pca)

y_pred_pca = xgb_model.predict(X_test_pca)
pca_rmse = mean_squared_error(y_test_pca, y_pred_pca, squared=False)
print(f"✅ PCA-transformed XGBoost RMSE: {pca_rmse:.4f}")


from rdkit import Chem
import deepchem as dc

# Load drug SMILES (you need a dataset with drug structures)
drug_smiles = pd.read_csv("gdsc_drug_smiles.csv")  # Example file with "DRUG_ID" and "SMILES"

# Featurize drugs into molecular fingerprints
featurizer = dc.feat.CircularFingerprint(size=2048)  # Generate 2048-bit molecular fingerprints
drug_features = {drug: featurizer.featurize(Chem.MolFromSmiles(smiles))[0] for drug, smiles in zip(drug_smiles["DRUG_ID"], drug_smiles["SMILES"])}

# Convert drug features into a DataFrame
drug_feature_df = pd.DataFrame.from_dict(drug_features, orient="index")

# Merge drug features with gene embeddings for model training
merged_features = gdsc_ic50_filtered.merge(drug_feature_df, on="DRUG_ID")

print(f"✅ Merged feature shape: {merged_features.shape}")
