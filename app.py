import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model
import folium
from streamlit_folium import st_folium
import joblib

# Cache resource-intensive tasks
@st.cache_resource
def load_models():
    scaler = joblib.load("models/scaler.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    mlp_model = load_model("models/mlp_model.h5")
    return scaler, label_encoder, mlp_model

# Main App
def main():
    st.title("Electricity Access and Microgrid Viability")
    
    # Load models
    try:
        scaler, label_encoder, mlp_model = load_models()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return

    # County selection
    try:
        counties = label_encoder.classes_
        selected_county = st.selectbox("Select a County", counties)
        encoded_county = label_encoder.transform([selected_county])[0]
    except Exception as e:
        st.error(f"Error with county selection: {e}")
        return

    # Simulated data input
    df_input = pd.DataFrame({
        "Pop_Density_2020": [100],
        "Wind_Speed": [5.0],
        "Latitude": [-1.2921],
        "Longitude": [36.8219],
        "Grid_Value": [0.8],
        "Cluster": [1],
        "Cluster_Mean_Pop_Density": [200],
        "Cluster_Mean_Wind_Speed": [5.5],
        "Income_Distribution_encoded": [encoded_county]
    })

    # Feature scaling
    try:
        X_scaled = scaler.transform(df_input)
    except ValueError as e:
        st.error(f"Feature mismatch: {e}")
        st.write("Expected columns:", scaler.feature_names_in_)
        st.write("Input columns:", df_input.columns.tolist())
        return

    # Model prediction
    try:
        predictions = mlp_model.predict([X_scaled, np.array([encoded_county])])
        st.write("Prediction:", predictions)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return

    # Visualization
    folium_map = folium.Map(location=[-1.2921, 36.8219], zoom_start=6)
    folium.Marker(location=[-1.2921, 36.8219], popup="Sample Point").add_to(folium_map)
    st_folium(folium_map, width=700)

if __name__ == "__main__":
    main()
