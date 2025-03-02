import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from data_preprocessing import prepare_data

# Load data
X, y = prepare_data()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define MLP Model
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

# Evaluate model
mlp_pred = mlp_model.predict(X_test).flatten()
mlp_rmse = mean_squared_error(y_test, mlp_pred, squared=False)

print(f"✅ MLP RMSE: {mlp_rmse:.4f}")
