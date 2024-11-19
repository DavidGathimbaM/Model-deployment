import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model
import folium
from streamlit_folium import st_folium
import joblib

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
        'Pop_Density_2020', 'Wind_Speed', 'Latitude', 'Longitude', 
        'Grid_Value', 'Cluster', 'Stability_Score', 'Income_Distribution_encoded', 'Cluster_Mean_Pop_Density', 'Cluster_Mean_Wind_Speed'
    ]
    missing_columns = [col for col in required_columns if col not in county_data.columns]
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        return

    # Align the columns for scaling with validation
    try:
        X_numeric = county_data[required_columns]
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
        predictions = mlp_model.predict([X_scaled, county_data['Income_Distribution_encoded']])
        county_data['Electricity_Predicted'] = (predictions > 0.5).astype(int)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return

    # Sidebar for viability parameters
    grid_proximity_threshold = st.sidebar.slider("Grid Proximity Threshold (km)", 1, 50, 5)
    wind_speed_threshold = st.sidebar.slider("Wind Speed Threshold (m/s)", 1, 15, 8)

    # Viability calculations
    county_data['Distance_to_Grid'] = county_data['Grid_Value'] * 10  # Example scaling for proximity

    # Determine viability
    county_data['Viability'] = np.where(
        (county_data['Electricity_Predicted'] == 0) & (county_data['Distance_to_Grid'] <= grid_proximity_threshold),
        "Viable for Grid Extension",
        np.where(
            (county_data['Electricity_Predicted'] == 0) & (county_data['Wind_Speed'] >= wind_speed_threshold),
            "Viable for Wind Microgrid",
            "Electrified"
        )
    )

    # Display viability results
    st.write("Viability Analysis:")
    st.dataframe(county_data[['Latitude', 'Longitude', 'Electricity_Predicted', 'Distance_to_Grid', 'Viability']])

    # Visualize on map
    try:
        st.write("Electrification and Viability Map:")
        folium_map = folium.Map(location=[county_data['Latitude'].mean(), county_data['Longitude'].mean()], zoom_start=7)
        for _, row in county_data.iterrows():
            if row['Viability'] == "Viable for Grid Extension":
                color = 'blue'
            elif row['Viability'] == "Viable for Wind Microgrid":
                color = 'purple'
            elif row['Electricity_Predicted'] == 1:
                color = 'green'
            else:
                color = 'red'
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=f"Viability: {row['Viability']}, Prediction: {'Electricity' if row['Electricity_Predicted'] == 1 else 'No Electricity'}"
            ).add_to(folium_map)

        # Add a legend to the map
        legend_html = """
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 120px; 
                    background-color: white; z-index:1000; font-size:14px;
                    border:2px solid grey; border-radius:8px; padding:10px;">
            <b>Map Legend</b><br>
            <i style="background: green; width: 10px; height: 10px; display: inline-block;"></i> Electrified<br>
            <i style="background: blue; width: 10px; height: 10px; display: inline-block;"></i> Grid Extension<br>
            <i style="background: purple; width: 10px; height: 10px; display: inline-block;"></i> Wind Microgrid<br>
            <i style="background: red; width: 10px; height: 10px; display: inline-block;"></i> No Electricity<br>
        </div>
        """
        folium_map.get_root().html.add_child(folium.Element(legend_html))

        st_folium(folium_map, width=700)
    except Exception as e:
        st.error(f"Error displaying map: {e}")

if __name__ == "__main__":
    main()
