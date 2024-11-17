import streamlit as st
import gdown
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Set Google Drive file ID for the CSV file
file_id = 'YOUR_FILE_ID'
download_url = f'https://drive.google.com/uc?id={file_id}'
output_file = 'data.csv'

# Load pre-trained models
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
hdbscan_model = joblib.load('hdbscan_model.pkl')

@st.cache_data  # Cache the data download and processing
def download_and_load_data(url):
    # Download file from Google Drive
    gdown.download(url, output_file, quiet=False)
    # Load CSV data into a DataFrame
    df = pd.read_csv(output_file)
    return df

@st.cache_data
def apply_models(df):
    # Preprocess: Select required numeric columns and one-hot encode Income_Distribution
    income_columns = [col for col in scaler.feature_names_in_ if col.startswith("Income_")]
    numeric_columns = ['Pop_Density_2020', 'Wind_Speed', 'Latitude', 'Longitude', 'Grid_Value']

    # Ensure all expected columns are present
    for col in income_columns:
        if col not in df.columns:
            df[col] = 0  # Add missing columns with default value 0

    # Combine numeric and one-hot encoded columns
    clustering_data = df[numeric_columns + income_columns]

    # Scale the numeric data using the pre-trained scaler
    clustering_data_scaled = scaler.transform(clustering_data)

    # Reduce dimensionality using pre-trained PCA
    clustering_data_reduced = pca.transform(clustering_data_scaled)
    df['PCA_Component_1'] = clustering_data_reduced[:, 0]
    df['PCA_Component_2'] = clustering_data_reduced[:, 1]

    # Apply pre-trained HDBSCAN model for clustering
    clusters = hdbscan_model.fit_predict(clustering_data_reduced)
    df['Cluster'] = clusters
    df['Stability_Score'] = hdbscan_model.probabilities_

    return df, clusters

# Streamlit app interface
st.title("Electrification Planning and Clustering Insights")
st.write("This app downloads a large dataset from Google Drive, processes it using pre-trained models, and provides electrification insights.")

# Step 1: Download and load data
st.write("Loading and processing data...")
try:
    # Load data from Google Drive
    data = download_and_load_data(download_url)
    st.write("Data loaded successfully!")

    # Step 2: Apply pre-trained models
    data, clusters = apply_models(data)
    st.write("Data processed and clustered successfully!")

    # Step 3: Dropdown selection for Income_Distribution
    st.write("Select a region (county):")
    region = st.selectbox("County", options=data['Income_Distribution'].unique())

    # Filter data for the selected county
    filtered_data = data[data['Income_Distribution'] == region]

    # Display clustering results
    st.write("Clustering Results for the Selected County:")
    st.write(filtered_data[['Latitude', 'Longitude', 'Pop_Density_2020', 'Cluster', 'Stability_Score']])

    # Visualize clusters
    st.write("Cluster Visualization:")
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(data['PCA_Component_1'], data['PCA_Component_2'], c=clusters, cmap='viridis', s=5)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('HDBSCAN Clustering Results')
    st.pyplot(fig)

    # Insights based on clustering
    st.write("**Insights:**")
    st.write("""
    - Low-density clusters with suitable wind conditions are good candidates for wind microgrids.
    - High-density clusters near existing grids may benefit from grid extensions.
    - Noise points (-1) may represent isolated or unique regions requiring further analysis.
    """)

except Exception as e:
    st.write("Error loading or processing data:", e)
