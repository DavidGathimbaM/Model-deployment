import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium
from tensorflow.keras.models import load_model
import hdbscan
from sklearn.preprocessing import StandardScaler
import os

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load models and scalers
@st.cache_resource
def load_models():
    mlp_model = load_model("models/mlp_model.h5")
    scaler = joblib.load("models/scaler.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    hdbscan_model = joblib.load("models/hdbscan_model.pkl")
    return mlp_model, scaler, label_encoder, hdbscan_model

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/final_df.csv")

# Main app function
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

    # Ensure required columns are present
    required_features = scaler.feature_names_in_
    st.write("Required features for prediction:", required_features)

    # Check and handle missing features
    missing_features = [feature for feature in required_features if feature not in county_data.columns]
    if missing_features:
        st.warning(f"Missing features: {missing_features}. Imputing default values.")
        for feature in missing_features:
            county_data[feature] = 0  # Default imputation value

    # Align features order with the scaler
    county_data = county_data[required_features]

    # Scale numeric features
    try:
        X_scaled = scaler.transform(county_data)
    except Exception as e:
        st.error(f"Error during scaling: {e}")
        st.stop()

    # HDBSCAN clustering insights
    st.write("HDBSCAN Clustering Insights:")
    cluster_labels = hdbscan_model.labels_
    st.write(f"Cluster labels: {cluster_labels}")

    # Predict electricity availability using the MLP model
    predictions = mlp_model.predict(X_scaled)
    county_data['Electricity_Predicted'] = (predictions > 0.5).astype(int)

    # Display predictions
    st.write("Predictions for Selected County:")
    st.dataframe(county_data[['Latitude', 'Longitude', 'Electricity_Predicted']])

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
