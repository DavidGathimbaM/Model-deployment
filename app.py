import streamlit as st
import numpy as np
import pandas as pd
import joblib
import folium
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import os
from streamlit_folium import st_folium
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU usage for TensorFlow

# Load Models and Scalers
@st.cache_resource
def load_models():
    try:
        mlp_model = load_model("models/mlp_model.h5")
        scaler = joblib.load("models/scaler.pkl")
        label_encoder = joblib.load("models/label_encoder.pkl")
        hdbscan_model = joblib.load("models/hdbscan_model.pkl")
        return mlp_model, scaler, label_encoder, hdbscan_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

# Load the dataset
@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/final_df.csv")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

# Main App
def main():
    st.title("Electrification Planning in Kenya")
    st.write("Explore electrification recommendations by county.")

    # Load models and data
    mlp_model, scaler, label_encoder, hdbscan_model = load_models()
    df = load_data()

    # Sidebar: User County Selection
    counties = df['Income_Distribution'].unique()
    selected_county = st.sidebar.selectbox("Select a County", counties)

    # Filter data for the selected county
    county_data = df[df['Income_Distribution'] == selected_county]

    # Validate data availability
    if county_data.empty:
        st.error(f"No data available for the selected county: {selected_county}")
        st.stop()

    # Strip whitespace from column names
    county_data.columns = county_data.columns.str.strip()

    # Check for Latitude and Longitude columns
    if 'Latitude' not in county_data.columns or 'Longitude' not in county_data.columns:
        st.error("Latitude and Longitude columns are missing from the dataset. Cannot display map.")
        st.stop()

    # Rename latitude and longitude for st.map compatibility
    county_data_map = county_data.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'})

    # Display map with latitude and longitude
    st.map(county_data_map[['latitude', 'longitude']])

    # Required columns for MLP prediction
    required_columns = [
        'Pop_Density_2020', 'Wind_Speed', 'Latitude', 'Longitude', 'Grid_Value',
        'Cluster', 'Cluster_Mean_Pop_Density', 'Cluster_Mean_Wind_Speed', 'Income_Distribution_encoded'
    ]

    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in county_data.columns]
    if missing_columns:
        st.error(f"The following required columns are missing: {missing_columns}")
        st.stop()

    # Prepare numeric features for prediction
    X_numeric = county_data[required_columns]

    # Align features with the scaler
    if hasattr(scaler, 'feature_names_in_'):
        required_features = scaler.feature_names_in_  # Features used during training
    else:
        required_features = required_columns  # Fallback to required columns if attribute is missing

    # Handle missing features for scaler
    for feature in required_features:
        if feature not in X_numeric.columns:
            X_numeric.loc[:, feature] = 0  # Add missing features as zeros

    # Ensure correct feature order
    X_numeric = X_numeric[required_features]

    # Standardize features
    X_numeric_scaled = scaler.transform(X_numeric)

    # Prepare encoded categorical features
    X_county_encoded = county_data['Income_Distribution_encoded'].values.reshape(-1, 1)

    # Make predictions with the correct input format
    try:
        predictions = mlp_model.predict([X_numeric_scaled, X_county_encoded])
        county_data.loc[:, 'Electricity_Predicted'] = (predictions > 0.5).astype(int)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.stop()

    # Visualization with Folium
    st.write("Electrification Map:")
    folium_map = folium.Map(location=[county_data['Latitude'].mean(), county_data['Longitude'].mean()], zoom_start=8)
    for _, row in county_data.iterrows():
        color = 'green' if row['Electricity_Predicted'] == 1 else 'red'
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=f"Prediction: {'Electricity' if row['Electricity_Predicted'] == 1 else 'No Electricity'}"
        ).add_to(folium_map)
    st_folium(folium_map, width=700)

if __name__ == "__main__":
    main()
