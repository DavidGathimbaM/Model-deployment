import os
import streamlit as st
import pandas as pd
import joblib
import folium
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from streamlit_folium import st_folium
import warnings

# Suppress TensorFlow and general warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Load models and scalers
@st.cache_resource
def load_models():
    try:
        mlp_model = load_model("models/mlp_model.h5")
        mlp_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        scaler = joblib.load("models/scaler.pkl")
        label_encoder = joblib.load("models/label_encoder.pkl")
        hdbscan_model = joblib.load("models/hdbscan_model.pkl")
        return mlp_model, scaler, label_encoder, hdbscan_model
    except Exception as e:
        st.error(f"Error loading models or scalers: {e}")
        st.stop()

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/final_df.csv")
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'data/final_df.csv' exists.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

# Main application
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

    if county_data.empty:
        st.error(f"No data available for the selected county: {selected_county}")
        st.stop()

    # Ensure Latitude and Longitude columns exist
    if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        st.error("Latitude or Longitude columns are missing. Cannot render the map.")
        st.stop()

    # HDBSCAN clustering
    try:
        pca_features = county_data[['PCA_Component_1', 'PCA_Component_2']]
        clusters = hdbscan_model.fit_predict(pca_features)
        county_data['Cluster'] = clusters
        county_data['Stability_Score'] = hdbscan_model.probabilities_
    except Exception as e:
        st.error(f"Error applying HDBSCAN clustering: {e}")
        st.stop()

    # Debugging: HDBSCAN results
    st.write("HDBSCAN clustering results:")
    st.dataframe(county_data[['Cluster', 'Stability_Score']])

    # Prepare inputs for the MLP model
    numeric_features = [
        'Pop_Density_2020', 'Wind_Speed', 'Latitude', 'Longitude', 'Grid_Value',
        'Cluster', 'Stability_Score', 'Income_Distribution_encoded',
        'Cluster_Mean_Pop_Density', 'Cluster_Mean_Wind_Speed'
    ]
    missing_features = [feature for feature in numeric_features if feature not in county_data.columns]
    if missing_features:
        st.error(f"Missing features: {missing_features}")
        st.stop()

    X_numeric = county_data[numeric_features]
    try:
        X_numeric_scaled = scaler.transform(X_numeric)
    except Exception as e:
        st.error(f"Error scaling numeric features: {e}")
        st.stop()

    X_county = county_data['Income_Distribution_encoded']
    try:
        predictions = mlp_model.predict([X_numeric_scaled, X_county])
        county_data['Electricity_Predicted'] = (predictions > 0.5).astype(int)
    except Exception as e:
        st.error(f"Error making predictions: {e}")
        st.stop()

    # Visualization with Folium
    st.write("Electrification Map:")
    folium_map = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=6)
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
