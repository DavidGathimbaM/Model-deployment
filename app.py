import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import hdbscan
import folium
from streamlit_folium import st_folium

# Load models and data
@st.cache_resource
def load_models():
    mlp_model = load_model("models/mlp_model.h5")
    scaler = joblib.load("models/scaler.pkl")
    hdbscan_model = joblib.load("models/hdbscan_model.pkl")
    return mlp_model, scaler, hdbscan_model

@st.cache_data
def load_data():
    return pd.read_csv("data/final_df.csv")

# Main app function
def main():
    st.title("Electrification Planning with HDBSCAN and MLP")
    st.write("Analyze electrification data and clustering insights.")

    # Load models and data
    mlp_model, scaler, hdbscan_model = load_models()
    df = load_data()

    # Sidebar selection
    counties = df['Income_Distribution'].unique()
    selected_county = st.sidebar.selectbox("Select a County", counties)

    # Filter data for the selected county
    county_data = df[df['Income_Distribution'] == selected_county].copy()

    # Ensure required columns exist
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

    # Extract categorical feature for embedding
    county_encoded = county_data['Income_Distribution_encoded'].values

    try:
        # HDBSCAN clustering insights
        if len(hdbscan_model.labels_) == len(df):
            # Match labels to the dataset
            df['Cluster_Labels'] = hdbscan_model.labels_
            county_data['Cluster_Labels'] = df.loc[county_data.index, 'Cluster_Labels']
        else:
            st.warning("HDBSCAN model does not match the current dataset size. Clustering insights will be skipped.")
            county_data['Cluster_Labels'] = -1  # Assign -1 for unknown clusters

        st.write("HDBSCAN Clustering Insights:")
        st.dataframe(county_data[['Cluster_Labels', 'Pop_Density_2020']])

        # MLP model predictions
        predictions = mlp_model.predict([X_scaled, county_encoded])
        county_data['Electricity_Predicted'] = (predictions > 0.5).astype(int)

        # Display predictions
        st.write(f"Predictions for {selected_county}:")
        st.dataframe(county_data[['Latitude', 'Longitude', 'Electricity_Predicted']])

        # Visualization with Folium
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
        st_folium(folium_map, width=700)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
