# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# import pickle

# # Load dataset
# df = pd.read_csv("dataset/train.csv")

# # Convert 'Garage' column from Yes/No to 1/0
# df['Garage'] = df['Garage'].map({'Yes': 1, 'No': 0})

# # Drop rows with missing values (optional but safer)
# df = df.dropna()

# # Select numeric features
# X = df[['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'Garage']]
# y = df['Price']

# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train the model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Save the model
# with open('model.pkl', 'wb') as f:
#     pickle.dump(model, f)

# print("Model trained successfully and saved as model.pkl")


import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Load dataset
df = pd.read_csv("dataset/train.csv")

# Drop Id column (not a feature)
df = df.drop(columns=["Id"])

# Identify categorical columns
categorical_cols = ["Location", "Condition", "Garage"]

# One-hot encode categorical columns (don't drop first to keep consistent feature columns)
df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

# Separate features and target
X = df.drop(columns="Price")
y = df["Price"]

# Save feature column names for later use
feature_columns = X.columns.tolist()
with open("feature_columns.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Predict on training data to evaluate
y_pred = model.predict(X_scaled)

# Metrics
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Training MSE: {mse:.2f}")
print(f"Training RMSE: {rmse:.2f}")
print(f"Training MAE: {mae:.2f}")
print(f"Training R2 Score: {r2:.4f}")

# Save the model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model training complete and files saved.")
