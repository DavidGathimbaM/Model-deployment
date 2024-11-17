import streamlit as st
import pandas as pd
import joblib
import folium
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import os
from streamlit_folium import st_folium
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU usage for TensorFlow

# Load Models and Scalers
@st.cache_resource
def load_models():
    try:
        mlp_model = load_model("models/mlp_model.h5")
        scaler = joblib.load("models/scaler.pkl")
        label_encoder = joblib.load("models/label_encoder.pkl")
        hdbscan_model = joblib.load("models/hdbscan_model.pkl")
        return mlp_model, scaler, label_encoder, hdbscan_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

# Load the dataset
@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/final_df.csv")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

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

    # Validate data availability
    if county_data.empty:
        st.error(f"No data available for the selected county: {selected_county}")
        st.stop()

    # Debugging: Display available columns
    st.write("Columns in county_data before processing:", county_data.columns.tolist())

    # Strip whitespace from column names
    county_data.columns = county_data.columns.str.strip()

    # Check for Latitude and Longitude columns
    if 'Latitude' not in county_data.columns or 'Longitude' not in county_data.columns:
        st.error("Latitude and Longitude columns are missing from the dataset. Cannot display map.")
        st.stop()

    # Rename latitude and longitude for st.map compatibility
    county_data_map = county_data.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'})

    # Debugging: Display renamed columns
    st.write("Columns in county_data_map:", county_data_map.columns.tolist())

    # Display map with latitude and longitude
    st.map(county_data_map[['latitude', 'longitude']])

    # Required columns for MLP prediction
    required_columns = [
        'Pop_Density_2020', 'Wind_Speed', 'Latitude', 'Longitude', 'Grid_Value',
        'Cluster', 'Cluster_Mean_Pop_Density', 'Cluster_Mean_Wind_Speed', 'Income_Distribution_encoded'
    ]

    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in county_data.columns]
    if missing_columns:
        st.error(f"The following required columns are missing: {missing_columns}")
        st.stop()

    # Select features for prediction
    X_numeric = county_data[required_columns]

    # Align features with the scaler
    if hasattr(scaler, 'feature_names_in_'):
        required_features = scaler.feature_names_in_  # Features used during training
    else:
        required_features = required_columns  # Fallback to required columns if attribute is missing
    st.write("Features seen during scaler fitting:", required_features)

    # Handle any missing features for scaler
    for feature in required_features:
        if feature not in X_numeric.columns:
            X_numeric[feature] = 0

    # Ensure correct feature order
    X_numeric = X_numeric[required_features]

    # Standardize features
    X_scaled = scaler.transform(X_numeric)

    # Make predictions
    X_county = county_data['Income_Distribution_encoded']
    predictions = mlp_model.predict(X_scaled, X_county)
    county_data['Electricity_Predicted'] = (predictions > 0.5).astype(int)

    # Debugging: Display predictions
    st.write("Updated county_data with predictions:")
    st.dataframe(county_data[['Latitude', 'Longitude', 'Electricity_Predicted']])

    # Visualization with Folium
    st.write("Electrification Map:")
    folium_map = folium.Map(location=[county_data['Latitude'].mean(), county_data['Longitude'].mean()], zoom_start=8)
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


# import streamlit as st
# import pandas as pd
# import joblib
# import folium
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import StandardScaler
# import os
# import warnings
# warnings.filterwarnings("ignore", category=SyntaxWarning)

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# # Load Models and Scalers
# @st.cache_resource
# def load_models():
#     mlp_model = load_model("models/mlp_model.h5")
#     scaler = joblib.load("models/scaler.pkl")
#     label_encoder = joblib.load("models/label_encoder.pkl")
#     hdbscan_model = joblib.load("models/hdbscan_model.pkl")
#     mlp_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#     return mlp_model, scaler, label_encoder, hdbscan_model

# # Load the dataset
# @st.cache_data
# def load_data():
#     return pd.read_csv("data/final_df.csv")

# # Main App
# def main():
#     st.title("Electrification Planning in Kenya")
#     st.write("Explore electrification recommendations by county.")

#     # Load models and data
#     mlp_model, scaler, label_encoder, hdbscan_model = load_models()
#     df = load_data()

#     # Sidebar: User County Selection
#     counties = df['Income_Distribution'].unique()
#     selected_county = st.sidebar.selectbox("Select a County", counties)

    
#     # Filter data for the selected county
#     county_data = df[df['Income_Distribution'] == selected_county]

#     # Strip whitespace from column names
#     county_data.columns = county_data.columns.str.strip()

#     # Rename latitude and longitude columns for st.map compatibility
#     # Ensure column names are lowercase for st.map
#     # county_data_map = df[df['Latitude', 'Longitude']==selected_county]
#     county_data_map = county_data.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'})


    
#     st.write(f"Showing data for {selected_county}")
#     # debug
#     st.write("Columns in county_data:", county_data.columns.tolist())
#     if 'Latitude' not in county_data.columns or 'Longitude' not in county_data.columns:
#         st.error("Latitude and Longitude columns are missing from the dataset. Cannot display map.")
#         st.stop()


#     st.map(county_data_map[['latitude', 'longitude']])

#     st.write("Columns in county_data before prediction:", county_data.columns.tolist())

#     # Define features used for prediction
#     numeric_features = [
#         'Pop_Density_2020', 'Wind_Speed', 'Latitude', 'Longitude', 'Grid_Value',
#         'Cluster', 'Stability_Score', 'Income_Distribution_encoded', 
#         'Cluster_Mean_Pop_Density', 'Cluster_Mean_Wind_Speed'
#     ]

#     # Apply HDBSCAN clustering to the data
#     st.write("Applying HDBSCAN clustering...")
#     pca_features = county_data[['PCA_Component_1', 'PCA_Component_2']]  # Replace with PCA features
#     clusters = hdbscan_model.fit_predict(pca_features)
#     county_data['Cluster'] = clusters
#     county_data['Stability_Score'] = hdbscan_model.probabilities_

#     # Debugging: Display HDBSCAN results
#     st.write("HDBSCAN clustering results:")
#     st.dataframe(county_data[['Cluster', 'Stability_Score']])

#     # List of required columns for the MLP model
#     required_columns = ['Pop_Density_2020', 'Wind_Speed', 'Latitude', 'Longitude', 'Grid_Value', 'Cluster', 'Cluster_Mean_Pop_Density', 'Cluster_Mean_Wind_Speed', 'Income_Distribution_encoded']
#     # Check if all required columns exist
#     missing_columns = [col for col in required_columns if col not in county_data.columns]
#     if missing_columns:
#     # Display an error in Streamlit if columns are missing
#         st.error(f"The following required columns are missing: {missing_columns}")
#         st.stop()
#     else:
#     # Select features if all required columns are present
#         # X_numeric = county_data[numeric_features]

#         # # Standardize features using the scaler
#         # X_scaled = scaler.transform(X_numeric)

#         # # Debugging: Check and align features with the scaler
#         # required_features = scaler.feature_names_in_  # Features used during training
#         # st.write("Features seen during scaler fitting:", required_features)
#         # st.write("Current features passed to scaler:", X_numeric.columns.tolist())
#         # Select and scale numeric features
#         X_numeric = county_data[numeric_features]
#         X_numeric_scaled = scaler.transform(X_numeric)

#         # Extract county categorical input
#         X_county = county_data['Income_Distribution_encoded']
#         # Debugging: Display shapes of inputs
#         st.write("Shape of X_numeric_scaled:", X_numeric_scaled.shape)
#         st.write("Shape of X_county:", X_county.shape)

#         # # Make predictions with two inputs
#         # predictions = mlp_model.predict([X_numeric_scaled, X_county])
#         # county_data['Electricity_Predicted'] = (predictions > 0.5).astype(int)

#         try:
#             predictions = mlp_model.predict([X_numeric_scaled, X_county])
#             county_data['Electricity_Predicted'] = (predictions > 0.5).astype(int)
#         except Exception as e:
#             st.error(f"Error during prediction: {e}")
#         return

#         # # Debugging: Ensure column exists
#         # st.write("Updated county_data with Electricity_Predicted column:")
#         # st.dataframe(county_data)

#         # Debugging: Display predictions
#         st.write("Updated county_data with predictions:")
#         st.dataframe(county_data[['Latitude', 'Longitude', 'Electricity_Predicted']])
#         # # Display predictions
#         # st.write("Predictions for Selected County:")
#         # st.dataframe(county_data[['Latitude', 'Longitude', 'Electricity_Predicted']])
#     # # Show Predictions
#     # X_numeric = county_data[['Pop_Density_2020', 'Wind_Speed', 'Latitude', 'Longitude', 'Grid_Value']]
#     # X_scaled = scaler.transform(X_numeric)
#     # predictions = mlp_model.predict(X_scaled)
#     # county_data['Electricity_Predicted'] = (predictions > 0.5).astype(int)

#     # st.write("Predictions for Selected County:")
#     # st.dataframe(county_data[['Latitude', 'Longitude', 'Electricity_Predicted']])

# #     # Visualization with Folium
# #     st.write("Electrification Map:")
# #     folium_map = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=6)
# #     for _, row in county_data.iterrows():
# #         color = 'green' if row['Electricity_Predicted'] == 1 else 'red'
# #         folium.CircleMarker(
# #             location=[row['Latitude'], row['Longitude']],
# #             radius=5,
# #             color=color,
# #             fill=True,
# #             fill_opacity=0.7,
# #             popup=f"Prediction: {'Electricity' if row['Electricity_Predicted'] == 1 else 'No Electricity'}"
# #         ).add_to(folium_map)
# #     st_folium = st_folium(folium_map, width=700)

# # if __name__ == "__main__":
# #     main()

#         # Visualization with Folium
#         st.write("Electrification Map:")
#         folium_map = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=6)
#         for _, row in county_data.iterrows():
#             color = 'green' if row['Electricity_Predicted'] == 1 else 'red'
#             folium.CircleMarker(
#                 location=[row['Latitude'], row['Longitude']],
#                 radius=5,
#                 color=color,
#                 fill=True,
#                 fill_opacity=0.7,
#                 popup=f"Prediction: {'Electricity' if row['Electricity_Predicted'] == 1 else 'No Electricity'}"
#             ).add_to(folium_map)
#         st_folium(folium_map, width=700)

# if __name__ == "__main__":
#     main()
