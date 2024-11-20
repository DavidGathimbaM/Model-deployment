import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model
import joblib
from geopy.distance import geodesic

@st.cache_resource
def load_models():
    """Load scaler, label encoder, and MLP model."""
    try:
        scaler = joblib.load("models/scaler.pkl")
        label_encoder = joblib.load("models/label_encoder.pkl")
        mlp_model = load_model("models/mlp_model.h5")
        return scaler, label_encoder, mlp_model
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.stop()

@st.cache_data
def load_data():
    """Load the dataset."""
    try:
        return pd.read_csv("data/final_df.csv")
    except FileNotFoundError as e:
        st.error(f"Dataset not found: {e}")
        st.stop()

def calculate_distance(row, user_coords):
    """Calculate geodesic distance for a given row."""
    return geodesic(user_coords, (row['Latitude'], row['Longitude'])).km

def main():
    st.title("Electricity Access and Microgrid Viability")
    st.write("Explore electricity access predictions and clustering insights.")
    
    # Load models and data
    scaler, label_encoder, mlp_model = load_models()
    df = load_data()

    # Sidebar for county selection
    counties = label_encoder.classes_
    selected_county = st.sidebar.selectbox("Select a County", counties)
    encoded_county = label_encoder.transform([selected_county])[0]

    # Filter data for selected county
    county_data = df[df['Income_Distribution_encoded'] == encoded_county]
    if county_data.empty:
        st.error("No data found for the selected county.")
        return

    st.write(f"Data for {selected_county}")
    st.dataframe(county_data)

    # Ensure required columns are present
    required_columns = [
        'Pop_Density_2020', 'Wind_Speed', 'Latitude', 'Longitude', 
        'Grid_Value', 'Cluster', 'Stability_Score', 'Income_Distribution_encoded', 'Cluster_Mean_Pop_Density', 'Cluster_Mean_Wind_Speed'
    ]
    missing_columns = [col for col in required_columns if col not in county_data.columns]
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        return

    # Align the columns for scaling
    X_numeric = county_data[required_columns]
    try:
        X_scaled = scaler.transform(X_numeric)
    except ValueError as e:
        st.error(f"Scaler feature mismatch: {e}")
        return

    # User Input for Latitude and Longitude
    st.sidebar.write("### Check Viability for a Specific Point")
    user_lat = st.sidebar.number_input("Enter Latitude", value=county_data['Latitude'].mean())
    user_lon = st.sidebar.number_input("Enter Longitude", value=county_data['Longitude'].mean())

    # Calculate nearest points for user input
    user_coords = (user_lat, user_lon)
    county_data['Distance_to_Point'] = county_data.apply(calculate_distance, axis=1, args=(user_coords,))
    nearest_points = county_data.nsmallest(2, 'Distance_to_Point')

    # Calculate mean wind speed and grid distance for the input point
    mean_wind_speed = nearest_points['Wind_Speed'].mean()
    grid_distance = nearest_points['Grid_Value'].iloc[0] * 10  # Convert Grid_Value scale to kilometers

    st.sidebar.write(f"**Mean Wind Speed (Nearest Points):** {mean_wind_speed:.2f} m/s")
    st.sidebar.write(f"**Distance to Nearest Grid:** {grid_distance:.2f} km")

    # Viability for User Input
    grid_proximity_threshold = 3  # in kilometers
    wind_speed_threshold = 6.5  # in m/s

    if grid_distance <= grid_proximity_threshold:
        viability = "Viable for Grid Extension"
    elif mean_wind_speed > wind_speed_threshold:
        viability = "Viable for Wind Microgrid"
    else:
        viability = "Not Viable"
    
    st.sidebar.write(f"### Viability for Input Point: {viability}")

    # Visualize on map
    st.write("Electrification and Viability Map:")
    try:
        folium_map = folium.Map(location=[county_data['Latitude'].mean(), county_data['Longitude'].mean()], zoom_start=7)

        # Add user-input point
        folium.Marker(
            location=[user_lat, user_lon],
            popup=(f"Input Point<br>Viability: {viability}<br>"
                   f"Mean Wind Speed: {mean_wind_speed:.2f} m/s<br>"
                   f"Distance to Grid: {grid_distance:.2f} km"),
            icon=folium.Icon(color="orange")
        ).add_to(folium_map)

        # Add county points
        for _, row in county_data.iterrows():
            if row['Distance_to_Grid'] <= grid_proximity_threshold:
                color = 'blue'
            elif row['Wind_Speed'] > wind_speed_threshold:
                color = 'purple'
            else:
                color = 'green' if row['Electricity_Predicted'] == 1 else 'red'

            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=(f"Viability: {row['Viability']}<br>"
                       f"Electricity: {'Yes' if row['Electricity_Predicted'] == 1 else 'No'}<br>"
                       f"Distance to Grid: {row['Distance_to_Grid']} km<br>"
                       f"Wind Speed: {row['Wind_Speed']} m/s")
            ).add_to(folium_map)

        st_folium(folium_map, width=700)
    except Exception as e:
        st.error(f"Error displaying map: {e}")

if __name__ == "__main__":
    main()
