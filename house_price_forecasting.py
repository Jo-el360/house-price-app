# house_price_forecasting.py

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------------
# 1. Load Dataset
# -----------------------------------
data = pd.read_csv('data.csv')  # Adjust path if needed

# Preview
print("Dataset Shape:", data.shape)
print(data.head())

# -----------------------------------
# 2. Preprocessing
# -----------------------------------

# Define target column
target_column = 'Price'  # <<< Change this if your target variable has a different name

# Separate features and target
X = data.drop(columns=[target_column])
y = data[target_column]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Categorical Features: {categorical_cols}")
print(f"Numerical Features: {numerical_cols}")

# Numeric preprocessing pipeline
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical preprocessing pipeline
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Full preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# -----------------------------------
# 3. Model - Smart Regression
# -----------------------------------

# Define base models
xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
lgbm_model = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)

# Stacking model
stacked_model = StackingRegressor(
    estimators=[
        ('xgb', xgb_model),
        ('lgbm', lgbm_model)
    ],
    final_estimator=Ridge(alpha=1.0),
    passthrough=True,
    n_jobs=-1
)

# Full pipeline (Preprocessing + Model)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', stacked_model)
])

# -----------------------------------
# 4. Train-Test Split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------------
# 5. Model Training
# -----------------------------------
model_pipeline.fit(X_train, y_train)

# -----------------------------------
# 6. Model Evaluation
# -----------------------------------
y_pred = model_pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"R2 Score: {r2:.2f}")

# -----------------------------------
# 7. Save the Model
# -----------------------------------
os.makedirs('saved_model', exist_ok=True)
joblib.dump(model_pipeline, 'saved_model/house_price_model.pkl')
print("✅ Model saved to 'saved_model/house_price_model.pkl'")

# -----------------------------------
# 8. Real-Time Prediction Function
# -----------------------------------

def predict_price(input_data: dict):
    """
    Predict house price from new input dictionary.

    Args:
    - input_data (dict): Input data as feature:value

    Returns:
    - prediction (float): Predicted house price
    """
    model = joblib.load('saved_model/house_price_model.pkl')
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    return prediction[0]

# Example for Real-Time Prediction
if __name__ == "__main__":
    # 👇 Replace the keys with your actual dataset feature names
    sample_input = {
        # 'Feature1': value,
        # 'Feature2': value,
        # ...
    }
    # Uncomment to test
    # print(f"Predicted House Price: ${predict_price(sample_input):,.2f}")
