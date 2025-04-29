import pandas as pd
import streamlit as st
import joblib

# Load model
model = joblib.load("saved_model/house_price_model.pkl")

st.title("🏠 House Price Prediction App")

# --- Single Input Prediction ---
st.header("🔍 Predict a Single House Price")

with st.form("prediction_form"):
    bedrooms = st.number_input("Bedrooms", min_value=0, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=0.0, step=0.5)
    sqft_living = st.number_input("Sqft Living", min_value=0)
    sqft_lot = st.number_input("Sqft Lot", min_value=0)
    floors = st.number_input("Floors", min_value=0.0, step=0.5)
    waterfront = st.selectbox("Waterfront", [0, 1])
    view = st.slider("View Score", 0, 4)
    condition = st.slider("Condition", 1, 5)
    sqft_above = st.number_input("Sqft Above", min_value=0)
    sqft_basement = st.number_input("Sqft Basement", min_value=0)
    yr_built = st.number_input("Year Built", min_value=1800, max_value=2025, step=1)
    yr_renovated = st.number_input("Year Renovated", min_value=0, max_value=2025, step=1)
    submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = pd.DataFrame([{
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "sqft_living": sqft_living,
            "sqft_lot": sqft_lot,
            "floors": floors,
            "waterfront": waterfront,
            "view": view,
            "condition": condition,
            "sqft_above": sqft_above,
            "sqft_basement": sqft_basement,
            "yr_built": yr_built,
            "yr_renovated": yr_renovated,
            # add other categorical values if needed
        }])
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Predicted Price: ${prediction:,.2f}")

# --- Batch Prediction ---
st.header("📂 Batch Prediction from CSV")

uploaded_file = st.file_uploader("Upload a CSV file with house features", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    try:
        predictions = model.predict(df)
        df["Predicted Price"] = predictions
        st.write(df)
        st.download_button(
            label="Download Predictions as CSV",
            data=df.to_csv(index=False),
            file_name="predicted_prices.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
