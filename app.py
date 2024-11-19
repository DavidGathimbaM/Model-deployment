import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import joblib

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Disable GPU for TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

@st.cache_resource
def load_models():
    """Load scaler, label encoder, and MLP model."""
    try:
        scaler = joblib.load("models/scaler.pkl")
        label_encoder = joblib.load("models/label_encoder.pkl")
        # Load the model without recompiling
        mlp_model = load_model("models/mlp_model.h5", compile=False)
        return scaler, label_encoder, mlp_model
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

@st.cache_data
def generate_map(data):
    """Generate a folium map without sampling."""
    folium_map = folium.Map(location=[data['Latitude'].mean(), data['Longitude'].mean()], zoom_start=7)
    marker_cluster = MarkerCluster().add_to(folium_map)
    
    for _, row in data.iterrows():
        color = (
            'blue' if row['Viability'] == "Viable for Grid Extension" else
            'purple' if row['Viability'] == "Viable for Wind Microgrid" else
            'green' if row['Electricity_Predicted'] == 1 else
            'red'
        )
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=f"Viability: {row['Viability']}"
        ).add_to(marker_cluster)
    
    return folium_map

@st.cache_data
def predict_with_model(model, inputs):
    """Run predictions using the loaded model."""
    return model.predict(inputs)

def main():
    st.title("Electricity Access and Microgrid Viability")
    st.write("Explore electricity access predictions and clustering insights.")

    # Load models and data
    try:
        scaler, label_encoder, mlp_model = load_models()
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
    if not county_mask.any():
        st.error("No data found for the selected county.")
        return

    st.write(f"Data for {selected_county}")
    st.dataframe(df.loc[county_mask])

    # Ensure required columns are present
    required_columns = [
        'Pop_Density_2020', 'Wind_Speed', 'Latitude', 'Longitude', 
        'Grid_Value', 'Cluster', 'Stability_Score', 'Income_Distribution_encoded', 'Cluster_Mean_Pop_Density', 'Cluster_Mean_Wind_Speed'
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        return

    # Align the columns for scaling with validation
    try:
        X_numeric = df.loc[county_mask, required_columns]
        if hasattr(scaler, 'feature_names_in_'):
            missing_features = set(scaler.feature_names_in_) - set(X_numeric.columns)
            if missing_features:
                st.error(f"Missing features required by the scaler: {missing_features}")
                return
        X_scaled = scaler.transform(X_numeric)
    except Exception as e:
        st.error(f"Error scaling features: {e}")
        return

    # Make predictions using the MLP model
    try:
        if county_mask.any():
            inputs = {
                "input_layer": X_scaled,
                "input_layer_1": df.loc[county_mask, 'Income_Distribution_encoded'].values.reshape(-1, 1)
            }
            predictions = predict_with_model(mlp_model, inputs)
            df.loc[county_mask, 'Electricity_Predicted'] = (predictions > 0.5).astype(int)
        else:
            st.error("No data available for predictions.")
            return
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return

    # Sidebar for viability parameters
    grid_proximity_threshold = st.sidebar.slider("Grid Proximity Threshold (km)", 1, 50, 5)
    wind_speed_threshold = st.sidebar.slider("Wind Speed Threshold (m/s)", 1, 15, 8)

    # Viability calculations
    try:
        df.loc[county_mask, 'Distance_to_Grid'] = df.loc[county_mask, 'Grid_Value'] * 10  # Example scaling for proximity

        # Determine viability
        df.loc[county_mask, 'Viability'] = np.where(
            (df.loc[county_mask, 'Electricity_Predicted'] == 0) & (df.loc[county_mask, 'Distance_to_Grid'] <= grid_proximity_threshold),
            "Viable for Grid Extension",
            np.where(
                (df.loc[county_mask, 'Electricity_Predicted'] == 0) & (df.loc[county_mask, 'Wind_Speed'] >= wind_speed_threshold),
                "Viable for Wind Microgrid",
                "Electrified"
            )
        )
    except Exception as e:
        st.error(f"Error during viability calculations: {e}")
        return

    # Display viability results
    st.write("Viability Analysis:")
    st.dataframe(df.loc[county_mask, ['Latitude', 'Longitude', 'Electricity_Predicted', 'Distance_to_Grid', 'Viability']])

    # Visualize on map
    try:
        st.write("Electrification and Viability Map:")
        map_data = df.loc[county_mask]
        folium_map = generate_map(map_data)
        st_folium(folium_map, width=700)
    except Exception as e:
        st.error(f"Error displaying map: {e}")

if __name__ == "__main__":
    main()
