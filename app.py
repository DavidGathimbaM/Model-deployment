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
    """Load scaler, label encoder, and MLP model."""
    scaler = joblib.load("models/scaler.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    mlp_model = load_model("models/mlp_model.h5")
    return scaler, label_encoder, mlp_model

@st.cache_data
def load_data():
    """Load the dataset."""
    return pd.read_csv("data/final_df.csv")

def main():
    st.title("Electricity Access and Microgrid Viability")
    st.write("Explore electricity access predictions and clustering insights.")
    
    # Load models and data
    try:
        scaler, label_encoder, mlp_model = load_models()
        df = load_data()
    except Exception as e:
        st.error(f"Error loading resources: {e}")
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
    county_data = df[df['Income_Distribution_encoded'] == encoded_county]
    if county_data.empty:
        st.error("No data found for the selected county.")
        return

    st.write(f"Data for {selected_county}")
    st.dataframe(county_data)

    # Ensure required columns are present
    required_columns = [
        'Pop_Density_2020', 'Wind_Speed', 'Latitude', 'Longitude', 'Grid_Value',
        'Cluster', 'Cluster_Mean_Pop_Density', 'Cluster_Mean_Wind_Speed', 
        'Income_Distribution_encoded'
    ]
    missing_columns = [col for col in required_columns if col not in county_data.columns]
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        return

    # Prepare input for scaling and prediction
    try:
        X_numeric = county_data[required_columns]
        X_scaled = scaler.transform(X_numeric)
    except ValueError as e:
        st.error(f"Feature mismatch: {e}")
        st.write("Expected columns:", scaler.feature_names_in_)
        st.write("Input columns:", county_data.columns.tolist())
        return

    # Make predictions using the MLP model
    try:
        predictions = mlp_model.predict([X_scaled, X_numeric['Income_Distribution_encoded']])
        county_data['Electricity_Predicted'] = (predictions > 0.5).astype(int)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return

    # Display predictions
    st.write("Predictions for Electricity Access:")
    st.dataframe(county_data[['Latitude', 'Longitude', 'Electricity_Predicted']])

    # Visualize on map
    try:
        st.write("Electrification Map:")
        folium_map = folium.Map(location=[county_data['Latitude'].mean(), county_data['Longitude'].mean()], zoom_start=7)
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
    except Exception as e:
        st.error(f"Error displaying map: {e}")

if __name__ == "__main__":
    main()
