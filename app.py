import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model
import pickle
from geopy.distance import geodesic

# Load pre-trained models and scaler
@st.cache_resource
def load_resources():
    try:
        scaler = pickle.load(open("models/scaler.pkl", "rb"))
        label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))
        mlp_model = load_model("models/mlp_model.h5")
    except FileNotFoundError as e:
        st.error(f"File not found: {e.filename}. Please upload the required model files.")
        st.stop()
    return scaler, label_encoder, mlp_model

scaler, label_encoder, mlp_model = load_resources()

# App title
st.title("Electrification Viability Analysis Tool")

# Automatically load dataset
@st.cache_resource
def load_dataset():
    # Replace the URL below with your GitHub raw dataset link
    # dataset_url = "https://raw.githubusercontent.com/your-github-username/your-repository/main/your-dataset.csv"
    try:
        df = pd.read_csv("data/final_df.csv")
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

df = load_dataset()

# Display dataset preview
st.write("Dataset Preview:")
st.dataframe(df.head())

# Columns required for analysis
required_columns = ['Pop_Density_2020', 'Wind_Speed', 'Latitude', 'Longitude', 
        'Grid_Value', 'Cluster', 'Stability_Score', 'Income_Distribution_encoded', 'Cluster_Mean_Pop_Density', 'Cluster_Mean_Wind_Speed']

# Ensure required columns are present
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error(f"The following required columns are missing: {', '.join(missing_columns)}")
else:
    # Extract required features and standardize them
    clustering_data = df[required_columns]
    clustering_data_scaled = scaler.transform(clustering_data)

    # Make predictions using the MLP model
    predictions = mlp_model.predict(clustering_data_scaled)
    df['Electricity_Predicted'] = (predictions > 0.5).astype(int)

    # User Input Section
    st.header("Input a Specific Location for Viability Analysis")

    latitude_input = st.number_input("Enter Latitude:", value=0.0, format="%.6f")
    longitude_input = st.number_input("Enter Longitude:", value=0.0, format="%.6f")

    # If user inputs both latitude and longitude
    if latitude_input != 0.0 and longitude_input != 0.0:
        # Calculate distance to grid for the input point
        distances = df.apply(
            lambda row: geodesic((latitude_input, longitude_input), (row['Latitude'], row['Longitude'])).km,
            axis=1
        )
        nearest_distance = distances.min()

        # Retrieve wind speed for the closest point in the dataset
        nearest_point = df.iloc[distances.idxmin()]
        wind_speed_at_point = nearest_point['Wind_Speed']

        # Decision Logic for Viability
        if nearest_distance > 3 and wind_speed_at_point > 6.5:
            viability = "Viable for Wind Microgrid"
        elif nearest_distance <= 3:
            viability = "Viable for Grid Extension"
        else:
            viability = "Not Viable"

        # Display Results
        st.write(f"### Viability Analysis for Location ({latitude_input}, {longitude_input}):")
        st.write(f"- **Nearest Grid Distance**: {nearest_distance:.2f} km")
        st.write(f"- **Wind Speed at Point**: {wind_speed_at_point:.2f} m/s")
        st.write(f"- **Viability**: {viability}")

        # Map Visualization for the Input Point
        st.write("Location and Viability Map:")
        folium_map = folium.Map(location=[latitude_input, longitude_input], zoom_start=10)

        # Add input point to the map
        folium.Marker(
            location=[latitude_input, longitude_input],
            popup=f"Viability: {viability}<br>Nearest Grid Distance: {nearest_distance:.2f} km<br>Wind Speed: {wind_speed_at_point:.2f} m/s",
            icon=folium.Icon(color="blue" if viability == "Viable for Grid Extension" else "green" if viability == "Viable for Wind Microgrid" else "red")
        ).add_to(folium_map)

        st_folium(folium_map, width=700)
