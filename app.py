# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import streamlit as st
import joblib
import os
try:
    import kagglehub
except ImportError:
    kagglehub = None
    print("⚠️ kagglehub not installed, please install with: pip install kagglehub")
# Load Dataset
# List of Indian states with approximate coordinates for demonstration
state_coords = {
    "Andhra Pradesh": (15.9129, 79.7400),
    "Arunachal Pradesh": (28.2180, 94.7278),
    "Assam": (26.2006, 92.9376),
    "Bihar": (25.0961, 85.3131),
    "Chhattisgarh": (21.2787, 81.8661),
    "Goa": (15.2993, 74.1240),
    "Gujarat": (22.2587, 71.1924),
    "Haryana": (29.0588, 76.0856),
    "Himachal Pradesh": (31.1048, 77.1734),
    "Jharkhand": (23.6102, 85.2799),
    "Karnataka": (15.3173, 75.7139),
    "Kerala": (10.8505, 76.2711),
    "Madhya Pradesh": (22.9734, 78.6569),
    "Maharashtra": (19.7515, 75.7139),
    "Manipur": (24.6637, 93.9063),
    "Meghalaya": (25.4670, 91.3662),
    "Mizoram": (23.1645, 92.9376),
    "Nagaland": (26.1584, 94.5624),
    "Odisha": (20.9517, 85.0985),
    "Punjab": (31.1471, 75.3412),
    "Rajasthan": (27.0238, 74.2179),
    "Sikkim": (27.5330, 88.5122),
    "Tamil Nadu": (11.1271, 78.6569),
    "Telangana": (18.1124, 79.0193),
    "Tripura": (23.9408, 91.9882),
    "Uttar Pradesh": (26.8467, 80.9462),
    "Uttarakhand": (30.0668, 79.0193),
    "West Bengal": (22.9868, 87.8550),
    "Delhi": (28.7041, 77.1025),
}

# Add a synthetic 'state' column to the dataset (randomly assigned for demonstration)
def load_data():
    if kagglehub:
        # Download dataset from Kaggle using kagglehub
        path = kagglehub.dataset_download("naiyakhalid/flood-prediction-dataset")
        for fname in ["flood.csv", "flood_data.csv"]:
            csv_file_path = os.path.join(path, fname)
            if os.path.exists(csv_file_path):
                break
        else:
            raise FileNotFoundError(f"No flood dataset found in {path}. Checked flood.csv and flood_data.csv.")
    else:
        csv_file_path = "flood_data.csv"

    # Load dataset
    df = pd.read_csv(csv_file_path)

    states = list(state_coords.keys())
    np.random.seed(42)
    df['state'] = np.random.choice(states, size=len(df))

    # Synthetic flood risk label
    df['flood_risk'] = np.where(
        (df['MonsoonIntensity'] + df['TopographyDrainage'] + df['RiverManagement']) > 15,
        'High', 'Low'
    )

    # Feature/target split
    X = df.drop(columns=['flood_risk'])
    y = df['flood_risk']

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Train Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X.drop(columns=['state']), y)

    return X, rf, le, df

X, rf, le, df = load_data()

# Streamlit App
st.title("🌊 Flood Risk Prediction using Machine Learning")
st.write("Enter environmental and socio-economic factors below to predict flood risk.")

# Streamlit UI for state selection
st.sidebar.header("Location Selection")
states = list(state_coords.keys())
selected_state = st.sidebar.selectbox("Select your state", states)

# Use mean values for sliders based on selected state
def get_state_means(state):
    state_data = df[df['state'] == state]
    means = {}
    for col in X.columns:
        if col != 'state':
            means[col] = int(state_data[col].mean()) if not state_data.empty else 5
    return means

means = get_state_means(selected_state)
user_inputs = []
for col in X.columns:
    if col != 'state':
        val = st.slider(col, 0, 10, means[col])
        user_inputs.append(val)

# Create input DataFrame
input_data = pd.DataFrame([user_inputs], columns=[col for col in X.columns if col != 'state'])

# Prediction button
if st.button("Predict Flood Risk"):
    prediction = rf.predict(input_data)
    predicted_label = le.inverse_transform(prediction)[0]
    if predicted_label == "High":
        st.error(f"⚠️ High Flood Risk Detected in {selected_state}!")
    else:
        st.success(f"✅ Low Flood Risk in {selected_state}")

# EDA and Visualization
if st.checkbox("Show Data Insights & Visualizations"):
    st.subheader("Dataset Preview")
    st.dataframe(X.head())

    st.subheader("Feature Distributions")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import io
    import base64
    fig, ax = plt.subplots(figsize=(15, 6))
    X.hist(ax=ax, bins=20)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

    st.subheader("Flood Risk Distribution")
    fig3, ax3 = plt.subplots()
    sns.countplot(x=le.inverse_transform(rf.predict(X)), ax=ax3)
    ax3.set_title("Flood Risk Distribution (Predicted)")
    st.pyplot(fig3)

# Map Visualization for selected state
if st.checkbox("Show High Flood Risk Locations on Map"):
    import folium
    from streamlit_folium import st_folium
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
    for state, (lat, lon) in state_coords.items():
        color = "red" if state == selected_state else "blue"
        folium.CircleMarker(
            location=[lat, lon],
            radius=10 if state == selected_state else 7,
            popup=f"{state}",
            color=color,
            fill=True,
            fill_color=color
        ).add_to(m)
    st_folium(m, width=700, height=500)
