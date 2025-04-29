
import streamlit as st
import pandas as pd
import joblib

st.title("🏠 House Price Prediction App")
st.write("Enter property details to get an instant price prediction.")

# Load trained model
model = joblib.load("saved_model/house_price_model.pkl")

# Input form
with st.form("predict_form"):
    bedrooms = st.number_input("Bedrooms", 0, 20, step=1)
    bathrooms = st.number_input("Bathrooms", 0.0, 20.0, step=0.5)
    sqft_living = st.number_input("Living Area (sqft)", 0, 10000)
    sqft_lot = st.number_input("Lot Size (sqft)", 0, 100000)
    floors = st.number_input("Floors", 0.0, 4.0, step=0.5)
    waterfront = st.selectbox("Waterfront View", [0, 1])
    view = st.slider("View Rating", 0, 4)
    condition = st.slider("Condition Rating", 1, 5)
    sqft_above = st.number_input("Sqft Above", 0, 10000)
    sqft_basement = st.number_input("Sqft Basement", 0, 5000)
    yr_built = st.number_input("Year Built", 1800, 2025)
    yr_renovated = st.number_input("Year Renovated", 0, 2025)
    date = st.text_input("Date", "2023-01-01")
    street = st.text_input("Street", "1234 Example St")
    city = st.text_input("City", "Seattle")
    statezip = st.text_input("StateZip", "WA 98103")
    country = st.text_input("Country", "USA")

    submitted = st.form_submit_button("Predict")

# Predict and display result
if submitted:
    input_data = {
        "bedrooms": bedrooms, "bathrooms": bathrooms,
        "sqft_living": sqft_living, "sqft_lot": sqft_lot,
        "floors": floors, "waterfront": waterfront, "view": view,
        "condition": condition, "sqft_above": sqft_above,
        "sqft_basement": sqft_basement, "yr_built": yr_built,
        "yr_renovated": yr_renovated, "date": date, "street": street,
        "city": city, "statezip": statezip, "country": country
    }
    prediction = model.predict(pd.DataFrame([input_data]))[0]
    st.success(f"🏷️ Estimated Price: ${prediction:,.2f}")
