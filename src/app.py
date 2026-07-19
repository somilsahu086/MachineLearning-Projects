import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# 1. Page Configuration & Title
st.set_page_config(page_title="California House Price Predictor", layout="centered")
st.title("🏡 California House Price Prediction App")
st.write("Enter the details below to estimate the median house value (in $100,000s).")

# 2. Load the Trained Model Safely
@st.cache_resource
def load_model():
    # Streamlit runs from root or src, so we handle path dynamically
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/house_price_model.pkl"))
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please run train.py first!")
        return None
    return joblib.load(model_path)

model = load_model()

if model is not None:
    st.subheader("📋 Input Features")
    
    # 3. Create Columns for Better Layout
    col1, col2 = st.columns(2)
    
    with col1:
        med_inc = st.number_input("Median Income (in $10,000s)", min_value=0.5, max_value=15.0, value=3.5, step=0.1)
        house_age = st.number_input("House Age (Years)", min_value=1.0, max_value=52.0, value=28.0, step=1.0)
        ave_rooms = st.number_input("Average Rooms per Dwelling", min_value=1.0, max_value=10.0, value=5.2, step=0.1)
        ave_bedrms = st.number_input("Average Bedrooms per Dwelling", min_value=0.5, max_value=5.0, value=1.1, step=0.1)
        
    with col2:
        population = st.number_input("Block Population", min_value=3.0, max_value=35000.0, value=1400.0, step=10.0)
        ave_occup = st.number_input("Average House Occupancy", min_value=1.0, max_value=10.0, value=3.0, step=0.1)
        latitude = st.number_input("Latitude (e.g., 32.0 to 42.0)", min_value=32.0, max_value=42.0, value=35.5, step=0.1)
        longitude = st.number_input("Longitude (e.g., -124.0 to -114.0)", min_value=-124.0, max_value=-114.0, value=-119.5, step=0.1)

    # 4. Feature Engineering (Same logic as preprocess.py)
    rooms_per_person = ave_rooms / ave_occup
    bedrooms_per_room = ave_bedrms / ave_rooms

    # 5. Prepare Input Data DataFrame with exact column names used in training
    input_data = pd.DataFrame([{
        'MedInc': med_inc,
        'HouseAge': house_age,
        'AveRooms': ave_rooms,
        'AveBedrms': ave_bedrms,
        'Population': population,
        'AveOccup': ave_occup,
        'Latitude': latitude,
        'Longitude': longitude,
        'RoomsPerPerson': rooms_per_person,
        'BedroomsPerRoom': bedrooms_per_room
    }])

    # 6. Prediction Button
    st.markdown("---")
    if st.button("🔮 Predict House Value", type="primary"):
        prediction = model.predict(input_data)[0]
        
        # Display results nicely
        st.balloons()
        st.success(f"### 🎉 Estimated Price: ${prediction * 100000:,.2f}")
        st.info(f"Raw Model Output (in hundred thousands): **{prediction:.4f}**")