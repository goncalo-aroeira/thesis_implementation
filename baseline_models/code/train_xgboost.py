import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from data_preprocessing import prepare_data
from utils import save_model, load_model

# Load data
X, y = prepare_data()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost Model
xgb_model = xgb.XGBRegressor(n_estimators=100, tree_method="hist", random_state=42)
xgb_model.fit(X_train, y_train)

# Save model
save_model(xgb_model, "/home/goncalo/scgpt/baseline_models/models/xgboost_trained_model.json")

# Evaluate model
y_pred = xgb_model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"✅ XGBoost RMSE: {rmse:.4f}")
