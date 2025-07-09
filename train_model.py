import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os

# Load the dataset
df = pd.read_csv("hardtail_bike_setup_dataset.csv")

# Feature columns and target columns
features = ["Height (in)", "Weight (lbs)", "Riding Style", "Terrain Type", "Skill Level"]
targets = [
    "Fork Pressure (psi)",
    "Front Tire Pressure (psi)",
    "Rear Tire Pressure (psi)",
    "Handlebar Width (mm)",
    "Estimated Saddle Height (in)"
]

# Define preprocessing for numeric and categorical features
categorical_features = ["Riding Style", "Terrain Type", "Skill Level"]
numeric_features = ["Height (in)", "Weight (lbs)"]

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="mean"), numeric_features),
    ("cat", OneHotEncoder(), categorical_features)
])

# Create a directory for saved models
os.makedirs("models", exist_ok=True)

# Train and save a model for each target
for target in targets:
    X = df[features]
    y = df[target]

    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, y_train)

    # Save the trained model
    filename = f"models/model_{target.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}.joblib"
    joblib.dump(model_pipeline, filename)
    print(f"✅ Saved model for '{target}' to {filename}")