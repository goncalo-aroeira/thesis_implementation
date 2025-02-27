import pandas as pd

# ===========================
# 1️⃣ Load & Merge GDSC Datasets
# ===========================

# Load GDSC1 and GDSC2
gdsc1 = pd.read_csv("gdsc/gdsc1_fitted_dose_response.csv")
gdsc2 = pd.read_csv("gdsc/gdsc2_fitted_dose_response.csv")

# Display first few rows
print(gdsc1.head())
print(gdsc2.head())

# Merge the datasets and remove duplicate rows
gdsc_merged = pd.concat([gdsc1, gdsc2], axis=0).drop_duplicates()

print(f"🔹 Merged dataset shape: {gdsc_merged.shape}")

# ===========================
# 2️⃣ Remove Non-Essential Columns
# ===========================

# Essential columns to keep
columns_to_keep = ["SANGER_MODEL_ID", "DRUG_ID", "LN_IC50"]

# Drop unnecessary columns
gdsc_cleaned = gdsc_merged[columns_to_keep]

print(f"🔹 Cleaned dataset shape: {gdsc_cleaned.shape}")

# ===========================
# 3️⃣ Confirm No Missing Values in LN_IC50
# ===========================

# Check if there are any missing values in LN_IC50
missing_ln_ic50 = gdsc_cleaned["LN_IC50"].isnull().sum()
print(f"\n📊 Missing Values in LN_IC50: {missing_ln_ic50}")

if missing_ln_ic50 > 0:
    # Fill missing IC50 values by averaging multiple entries for the same drug-cell pair
    gdsc_cleaned["LN_IC50"] = gdsc_cleaned.groupby(["SANGER_MODEL_ID", "DRUG_ID"])["LN_IC50"].transform(lambda x: x.fillna(x.mean()))

    # Fill remaining missing values with zero
    gdsc_cleaned.fillna(0, inplace=True)

# ===========================
# 4️⃣ Save Final Cleaned Dataset
# ===========================

# Save final cleaned dataset
gdsc_cleaned.to_csv("gdsc/gdsc_final_cleaned.csv", index=False)

print("✅ Final merged and cleaned GDSC dataset saved!")
print(f"📌 Final dataset shape: {gdsc_cleaned.shape}")
