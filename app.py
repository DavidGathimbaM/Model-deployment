import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import hdbscan
import folium
from streamlit_folium import st_folium

@st.cache_resource
def load_models():
    # Load models and scalers
    mlp_model = load_model("models/mlp_model.h5")
    mlp_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])  # Recompile to fix metrics warning
    scaler = joblib.load("models/scaler.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    hdbscan_model = joblib.load("models/hdbscan_model.pkl")
    return mlp_model, scaler, label_encoder, hdbscan_model

@st.cache_data
def load_data():
    return pd.read_csv("data/final_df.csv")

def main():
    st.title("Electrification Planning with HDBSCAN and MLP")
    st.write("Analyze electrification data and clustering insights.")

    # Load models and data
    mlp_model, scaler, label_encoder, hdbscan_model = load_models()
    df = load_data()

    # Sidebar selection
    counties = df['Income_Distribution'].unique()
    selected_county = st.sidebar.selectbox("Select a County", counties)

    # Filter data for the selected county
    county_data = df[df['Income_Distribution'] == selected_county].copy()

    # Ensure the label-encoded column exists
    if 'Income_Distribution_encoded' not in df.columns:
        st.error("The dataset does not contain the label-encoded column for Income Distribution.")
        st.stop()

    # Prepare encoded county column
    county_encoded = county_data['Income_Distribution_encoded'].values.reshape(-1, 1)

    # Define required columns for prediction
    required_columns = [
        'Pop_Density_2020', 'Wind_Speed', 'Latitude', 'Longitude', 'Grid_Value',
        'Cluster', 'Stability_Score', 'Income_Distribution_encoded', 'Cluster_Mean_Pop_Density', 'Cluster_Mean_Wind_Speed'
    ]
    missing_columns = [col for col in required_columns if col not in county_data.columns]
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        st.stop()

    # Prepare inputs for prediction
    X_numeric = county_data[required_columns]
    X_scaled = scaler.transform(X_numeric)

    try:
        # Predictions
        predictions = mlp_model.predict([X_scaled, county_encoded])
        county_data['Electricity_Predicted'] = (predictions > 0.5).astype(int)

        # Display predictions
        st.write(f"Predictions for {selected_county}:")
        st.dataframe(county_data[['Latitude', 'Longitude', 'Electricity_Predicted']])

        # Folium map
        st.write("Map with Predictions:")
        folium_map = folium.Map(location=[county_data['Latitude'].mean(), county_data['Longitude'].mean()], zoom_start=6)
        for _, row in county_data.iterrows():
            color = 'green' if row['Electricity_Predicted'] == 1 else 'red'
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=f"Electricity: {'Yes' if row['Electricity_Predicted'] == 1 else 'No'}"
            ).add_to(folium_map)
        st_folium(folium_map, width=700, height=500)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
