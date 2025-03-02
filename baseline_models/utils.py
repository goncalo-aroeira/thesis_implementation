import xgboost as xgb
import pickle

def save_model(model, filename):
    """Saves an XGBoost model to a file."""
    model.save_model(filename)
    print(f"✅ Model saved: {filename}")

def load_model(filename):
    """Loads a saved XGBoost model."""
    model = xgb.XGBRegressor()
    model.load_model(filename)
    print(f"✅ Model loaded: {filename}")
    return model
