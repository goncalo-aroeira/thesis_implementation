from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from data_preprocessing import prepare_data
from utils import save_model, load_model


# Load data
X, y = prepare_data()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save model
save_model(rf_model, "/home/goncalo/scgpt/baseline_models/models/rf_trained_model.json")

# Evaluate model
rf_pred = rf_model.predict(X_test)
rf_rmse = mean_squared_error(y_test, rf_pred, squared=False)

print(f"✅ Random Forest RMSE: {rf_rmse:.4f}")
