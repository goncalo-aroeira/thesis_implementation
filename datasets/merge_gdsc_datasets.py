import pandas as pd

# Load GDSC1 and GDSC2
gdsc1 = pd.read_csv("gdsc1_fitted_dose_response.csv")
gdsc2 = pd.read_csv("gdsc2_fitted_dose_response.csv")

# Display first few rows
print(gdsc1.head())
print(gdsc2.head())

# Check for common drugs and cell lines
common_drugs = set(gdsc1["DRUG_NAME"]) & set(gdsc2["DRUG_NAME"])
common_cells = set(gdsc1["SANGER_MODEL_ID"]) & set(gdsc2["SANGER_MODEL_ID"])

print(f"Number of common drugs: {len(common_drugs)}")
print(f"Number of common cell lines: {len(common_cells)}")

# Check common drug-cell pairs
common_pairs = gdsc1.merge(gdsc2, on=["SANGER_MODEL_ID", "DRUG_ID"], how="inner")
print(f"Number of common drug-cell pairs: {common_pairs.shape[0]}")

# Stack the datasets and drop duplicate rows
gdsc_merged = pd.concat([gdsc1, gdsc2], axis=0).drop_duplicates()

# Save the merged dataset
gdsc_merged.to_csv("gdsc_merged.csv", index=False)
print("Merged GDSC datasets by stacking all drug-cell pairs.")

# Check missing values
print(gdsc_merged.isnull().sum())

# Fill missing IC50 values by taking the mean if multiple values exist for the same drug-cell pair
gdsc_merged["LN_IC50"] = gdsc_merged.groupby(["SANGER_MODEL_ID", "DRUG_ID"])["LN_IC50"].transform(lambda x: x.fillna(x.mean()))

# Fill remaining missing values with zero (alternative: drop rows with missing critical data)
gdsc_merged.fillna(0, inplace=True)

# Save final cleaned dataset
gdsc_merged.to_csv("gdsc_final_cleaned.csv", index=False)
print("Final merged and cleaned GDSC dataset saved!")

# Check the final number of unique drug-cell pairs
unique_pairs = gdsc_merged.groupby(["SANGER_MODEL_ID", "DRUG_ID"]).size()
print(f"Total unique drug-cell pairs: {len(unique_pairs)}")

# Check shape
print("Final dataset shape:", gdsc_merged.shape)
