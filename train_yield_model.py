"""
Trains a RandomForestRegressor to predict crop yield (quintal/hectare)
from weather, soil, and input-use features. Saves the trained pipeline
(encoders + model) to yield_model.pkl for use in the Streamlit app.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

df = pd.read_csv("yield_data.csv")

target = "yield_quintal_per_hectare"
categorical_features = ["state", "crop", "season", "soil_type"]
numeric_features = [
    "year", "rainfall_mm", "avg_temp_c", "fertilizer_kg_per_ha",
    "pesticide_l_per_ha", "irrigation_coverage_pct", "area_hectare",
]

X = df[categorical_features + numeric_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ],
    remainder="passthrough",
)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)

pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
pipe.fit(X_train, y_train)

pred = pipe.predict(X_test)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print(f"MAE:  {mae:.2f} quintal/ha")
print(f"RMSE: {rmse:.2f} quintal/ha")
print(f"R2:   {r2:.3f}")

with open("yield_model.pkl", "wb") as f:
    pickle.dump({
        "pipeline": pipe,
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2},
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "options": {
            "state": sorted(df["state"].unique().tolist()),
            "crop": sorted(df["crop"].unique().tolist()),
            "season": sorted(df["season"].unique().tolist()),
            "soil_type": sorted(df["soil_type"].unique().tolist()),
        },
    }, f)

print("Saved yield_model.pkl")
