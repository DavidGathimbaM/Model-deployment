import streamlit as st
import pandas as pd
import joblib
import folium
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load Models and Scalers
@st.cache_resource
def load_models():
    mlp_model = load_model("models/mlp_model.h5")
    scaler = joblib.load("models/scaler.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    hdbscan_model = joblib.load("models/hdbscan_model.pkl")
    return mlp_model, scaler, label_encoder, hdbscan_model

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/final_merged_df.csv")

# Main App
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
    # Rename latitude and longitude columns for st.map compatibility
    county_data = county_data.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'})

    
    st.write(f"Showing data for {selected_county}")
    st.map(county_data[['latitude', 'longitude']])

    st.write("Columns in county_data before prediction:", county_data.columns.tolist())

    # List of required columns for the MLP model
    required_columns = ['Pop_Density_2020', 'Wind_Speed', 'Latitude', 'Longitude', 'Grid_Value']
    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in county_data.columns]
    if missing_columns:
    # Display an error in Streamlit if columns are missing
        st.error(f"The following required columns are missing: {missing_columns}")
        st.stop()
    else:
    # Select features if all required columns are present
        X_numeric = county_data[required_columns]

        # Standardize features using the scaler
        X_scaled = scaler.transform(X_numeric)

        # Make predictions using the MLP model
        predictions = mlp_model.predict(X_scaled)
        
        county_data['Electricity_Predicted'] = (predictions > 0.5).astype(int)

        # Debugging: Ensure column exists
        st.write("Updated county_data with Electricity_Predicted column:")
        st.dataframe(county_data)
        # Display predictions
        st.write("Predictions for Selected County:")
        st.dataframe(county_data[['Latitude', 'Longitude', 'Electricity_Predicted']])
    # # Show Predictions
    # X_numeric = county_data[['Pop_Density_2020', 'Wind_Speed', 'Latitude', 'Longitude', 'Grid_Value']]
    # X_scaled = scaler.transform(X_numeric)
    # predictions = mlp_model.predict(X_scaled)
    # county_data['Electricity_Predicted'] = (predictions > 0.5).astype(int)

    # st.write("Predictions for Selected County:")
    # st.dataframe(county_data[['Latitude', 'Longitude', 'Electricity_Predicted']])

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
    st_folium = st_folium(folium_map, width=700)

if __name__ == "__main__":
    main()
