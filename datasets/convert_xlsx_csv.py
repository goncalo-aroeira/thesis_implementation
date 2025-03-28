import pandas as pd
import polars as pl

# Step 1: Load CSV with better type inference
df = pd.read_csv("sc_data/rnaseq_fpkm.csv", low_memory=False)

# Step 2: Drop unnamed or empty columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Step 3: Convert all non-ID columns to numeric, forcing errors to NaN
for col in df.columns:
    if col != "model_id":
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Optional: show how many NaNs got introduced
print("🔍 NaNs per column (top 5):")
print(df.isna().sum().sort_values(ascending=False).head())

# Step 4: Convert to Polars and save as clean Parquet
pl_df = pl.from_pandas(df)
pl_df.write_parquet("sc_data/rnaseq_fpkm.parquet")

print("✅ Clean Parquet file saved as sc_data/rnaseq_fpkm.parquet")
