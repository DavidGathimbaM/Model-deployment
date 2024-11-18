/import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model
import pickle

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
st.title("Electrification and Viability Analysis")

# Automatically load dataset from GitHub
@st.cache_resource
def load_dataset():
    # Replace the URL below with your GitHub raw dataset link
    dataset_url = "data/final_df.csv"
    try:
        df = pd.read_csv(dataset_url)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

df = load_dataset()

# Display dataset preview
st.write("Dataset Preview:")
st.dataframe(df.head())

# Columns required for analysis
required_columns = [
    'Pop_Density_2020', 'Wind_Speed', 'Latitude', 'Longitude', 
    'Grid_Value', 'Cluster', 'Stability_Score', 'Income_Distribution_encoded', 'Cluster_Mean_Pop_Density', 'Cluster_Mean_Wind_Speed'
]

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

    # Calculate distance to grid (Example: using dummy distances for this implementation)
    # Replace this with actual distance calculation logic
    df['Distance_to_Grid'] = np.random.randint(1, 50, df.shape[0])  # Example: Random distances in km

    # Viability thresholds
    grid_proximity_threshold = 5  # Example: 5 km
    wind_speed_threshold = 4.5  # Example: 4.5 m/s

    # Classify areas based on electrification and viability
    df['Viability'] = np.where(
        df['Electricity_Predicted'] == 1,
        "Electrified",
        np.where(
            (df['Electricity_Predicted'] == 0) & (df['Distance_to_Grid'] <= grid_proximity_threshold),
            "Viable for Grid Extension",
            np.where(
                (df['Electricity_Predicted'] == 0) & (df['Distance_to_Grid'] > grid_proximity_threshold) & (df['Wind_Speed'] >= wind_speed_threshold),
                "Viable for Wind Microgrid",
                "Not Viable"
            )
        )
    )

    # Display the dataset with viability analysis
    st.write("Viability Analysis:")
    st.dataframe(df[['Latitude', 'Longitude', 'Electricity_Predicted', 'Distance_to_Grid', 'Viability']])

    # Map visualization
    try:
        st.write("Electrification and Viability Map:")
        folium_map = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=7)

        for _, row in df.iterrows():
            if row['Viability'] == "Electrified":
                color = 'green'
            elif row['Viability'] == "Viable for Grid Extension":
                color = 'blue'
            elif row['Viability'] == "Viable for Wind Microgrid":
                color = 'purple'
            else:  # Not Viable
                color = 'red'

            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=(
                    f"Viability: {row['Viability']}<br>"
                    f"Electricity Prediction: {'Electricity' if row['Electricity_Predicted'] == 1 else 'No Electricity'}<br>"
                    f"Distance to Grid: {row['Distance_to_Grid']} km<br>"
                    f"Wind Speed: {row['Wind_Speed']} m/s"
                )
            ).add_to(folium_map)

        st_folium(folium_map, width=700)
    except Exception as e:
        st.error(f"Error displaying map: {e}")
