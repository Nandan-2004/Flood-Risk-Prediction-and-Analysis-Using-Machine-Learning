# 🌊 Flood Risk Prediction and Analysis Using Machine Learning  

## 📌 Project Overview
This project demonstrates how **AI/ML can support Climate Risk & Disaster Management** by predicting **flood risk** across Indian states.  
It uses socio-economic and environmental indicators to classify states into **High** or **Low flood risk**.  
The solution integrates a **Streamlit web application** with:  
- Real-time flood risk prediction  
- State-wise selection with sliders  
- Exploratory Data Analysis (EDA) & visualizations  
- **Interactive map visualization** of Indian states  

---

## 📂 Dataset
- **Source:** Kaggle – [Flood Prediction Dataset](https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-dataset)  
- **Description:** Contains 50,000 records with numerical features representing flood factors:  
  - Monsoon Intensity  
  - Topography Drainage  
  - River Management  
  - Urbanization  
  - Deforestation  
  - Climate Change  
  - Agricultural Practices, etc.  

A **synthetic target label (`flood_risk`)** was generated using critical features.  
Additionally, a **synthetic `state` column** was added with randomly assigned Indian states for demonstration purposes.  

---

## 🛠️ Tools & Libraries
- **Programming Language:** Python  
- **Core Libraries:**  
  - `pandas`, `numpy` → Data preprocessing  
  - `matplotlib`, `seaborn` → Data visualization  
  - `scikit-learn` → Machine Learning (Random Forest Classifier, Logistic Regression)  
  - `streamlit` → Web application interface  
  - `folium`, `streamlit-folium` → Interactive map plotting  
  - `kagglehub` → Kaggle dataset downloader  

---

## 🚀 Features
1. **Flood Risk Prediction**  
   - Select a **state** from the sidebar  
   - Adjust feature sliders (default = mean values of that state)  
   - Predict **High ⚠️** or **Low ✅** flood risk  

2. **Exploratory Data Analysis (EDA)**  
   - Dataset preview  
   - Feature distributions (histograms)  
   - Correlation heatmap  
   - Predicted flood risk distribution  

3. **State-wise Interactive Map**  
   - Shows all Indian states as markers  
   - Selected state is highlighted in **red**  
   - Other states shown in **blue**  

---

