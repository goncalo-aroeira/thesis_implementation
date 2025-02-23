import pandas as pd

# Load Excel file
gdsc_data = pd.read_excel("GDSC2_fitted_dose_response_27Oct23.xlsx")

# Save as CSV
gdsc_data.to_csv("gdsc2_fitted_dose_response.csv", index=False)

print("GDSC file converted to CSV successfully!")
