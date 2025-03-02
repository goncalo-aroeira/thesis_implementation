from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from data_preprocessing import prepare_data

# Load data
X, y = prepare_data()

# Apply PCA (reduce dimensions)
pca = PCA(n_components=100)
X_pca = pca.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train XGBoost
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# Evaluate Model
y_pred_pca = xgb_model.predict(X_test)
pca_rmse = mean_squared_error(y_test, y_pred_pca, squared=False)

print(f"✅ PCA-transformed XGBoost RMSE: {pca_rmse:.4f}")
