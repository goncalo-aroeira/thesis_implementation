import os

print("✅ Running XGBoost...")
os.system("python train_xgboost.py")

print("✅ Running Random Forest...")
os.system("python train_random_forest.py")

print("✅ Running MLP...")
os.system("python train_mlp.py")
