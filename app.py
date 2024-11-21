import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import folium
from streamlit_folium import st_folium
import joblib
from geopy.distance import geodesic

@st.cache_resource
def load_models():
    """Load scaler and label encoder."""
    try:
        scaler = joblib.load("models/scaler.pkl")
        label_encoder = joblib.load("models/label_encoder.pkl")
        return scaler, label_encoder
    except Exception as e:
        st.error(f"Error loading models: {e}")
        raise

@st.cache_data
def load_data():
    """Load the dataset."""
    try:
        return pd.read_csv("data/final_df.csv")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        raise

def calculate_distance(row, user_coords):
    """Calculate geodesic distance for a given row."""
    return geodesic(user_coords, (row['Latitude'], row['Longitude'])).km

def main():
    st.title("Electricity Access and Microgrid Viability with Clustering")
    st.write("Explore clustering insights and determine viability for specific locations.")

    # Load models and data
    try:
        scaler, label_encoder = load_models()
        df = load_data()
    except Exception:
        return

    # Sidebar for county selection
    try:
        counties = label_encoder.classes_
        selected_county = st.sidebar.selectbox("Select a County", counties)
        encoded_county = label_encoder.transform([selected_county])[0]
    except Exception as e:
        st.error(f"Error with county selection: {e}")
        return

    # Filter data for selected county
    county_mask = df['Income_Distribution_encoded'] == encoded_county
    county_data = df.loc[county_mask]
    if county_data.empty:
        st.error("No data found for the selected county.")
        return

    st.write(f"Data for {selected_county}")
    st.dataframe(county_data)

    # Latitude and longitude bounds for the selected county
    lat_min, lat_max = county_data['Latitude'].min(), county_data['Latitude'].max()
    lon_min, lon_max = county_data['Longitude'].min(), county_data['Longitude'].max()

    # Sidebar for user input of latitude and longitude
    st.sidebar.write("### Enter Coordinates for a Specific Point")
    user_lat = st.sidebar.number_input("Enter Latitude", value=county_data['Latitude'].mean())
    user_lon = st.sidebar.number_input("Enter Longitude", value=county_data['Longitude'].mean())

    # Validate user input against county bounds
    if not (lat_min <= user_lat <= lat_max and lon_min <= user_lon <= lon_max):
        st.sidebar.error(f"Input coordinates are outside the bounds of {selected_county}.")
        st.sidebar.write(f"Allowed Latitude: {lat_min:.2f} to {lat_max:.2f}")
        st.sidebar.write(f"Allowed Longitude: {lon_min:.2f} to {lon_max:.2f}")
        return

    # Calculate the distances to user input and find the two closest points
    user_coords = (user_lat, user_lon)
    county_data['Distance_to_User'] = county_data.apply(calculate_distance, axis=1, args=(user_coords,))
    nearest_points = county_data.nsmallest(2, 'Distance_to_User')

    # Calculate mean wind speed and grid value for nearest points
    mean_wind_speed = nearest_points['Wind_Speed'].mean()
    nearest_grid_value = nearest_points['Grid_Value'].min()

    # Estimate distance to grid
    grid_distance = nearest_grid_value * 10  # Convert grid value scale to kilometers

    # Display results for user input
    st.sidebar.write(f"**Mean Wind Speed (Nearest Points):** {mean_wind_speed:.2f} m/s")
    st.sidebar.write(f"**Distance to Nearest Grid:** {grid_distance:.2f} km")

    # Determine viability for the user-input point
    grid_proximity_threshold = 20  # in kilometers
    wind_speed_threshold = 6.0  # in m/s

    if grid_distance <= grid_proximity_threshold:
        viability = "Viable for Grid Extension"
    elif mean_wind_speed > wind_speed_threshold:
        viability = "Viable for Wind Microgrid"
    else:
        viability = "Not Viable"

    st.sidebar.write(f"### Viability for Input Point: {viability}")

    # Visualize clusters and user input point on the map
    try:
        # Initialize map with clusters and user input point
        lat_mean = county_data['Latitude'].mean()
        lon_mean = county_data['Longitude'].mean()
        folium_map = folium.Map(location=[lat_mean, lon_mean], zoom_start=8)

        # Add cluster points
        for _, row in county_data.iterrows():
            color = 'blue' if row['Cluster'] == 1 else 'red' if row['Cluster'] == 2 else 'green'
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=(f"Cluster: {row['Cluster']}<br>"
                       f"Wind Speed: {row['Wind_Speed']:.2f} m/s<br>"
                       f"Grid Value: {row['Grid_Value']}<br>"
                       f"Distance to Grid: {row['Grid_Value'] * 10:.2f} km")
            ).add_to(folium_map)

        # Add user-input point as a marker
        folium.Marker(
            location=[user_lat, user_lon],
            popup=(f"Input Point<br>Viability: {viability}<br>"
                   f"Mean Wind Speed: {mean_wind_speed:.2f} m/s<br>"
                   f"Distance to Grid: {grid_distance:.2f} km"),
            icon=folium.Icon(color="orange")
        ).add_to(folium_map)

        st.write("Location and Cluster Map:")
        st_folium(folium_map, width=700)
    except Exception as e:
        st.error(f"Error displaying map: {e}")

if __name__ == "__main__":
    main()
