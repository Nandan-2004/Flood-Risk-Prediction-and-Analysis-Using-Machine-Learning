# Flood Risk Prediction and Analysis Using Machine Learning

## Project Overview
This project focuses on predicting flood risk based on environmental and socio-economic factors using Machine Learning techniques. The main objective is to assess flood-prone areas and provide insights for climate risk and disaster management.  

This repository contains milestones for **Week 2 (EDA & Preprocessing)** and **Week 3 (Model Training & Evaluation)**.

---

## Dataset
- **Source:** Kaggle Flood Prediction Dataset  
- **Link:** [Flood Prediction Dataset](https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-dataset)  
- **Description:** The dataset contains 50,000 records with numeric features representing flood-related factors, such as Monsoon Intensity, River Management, Urbanization, Deforestation, and more.

---

## Week 2 Milestone Tasks
1. **Data Collection:** Downloaded the Kaggle dataset and extracted CSV files.  
2. **Data Preprocessing:**  
   - Checked and handled missing values for numeric and categorical columns.  
   - Normalized numeric features where necessary.  
3. **Exploratory Data Analysis (EDA):**  
   - Plotted feature distributions to understand the spread of flood risk factors.  
   - Generated correlation heatmap to identify relationships between variables.  
   - Created a synthetic target variable (`flood_risk`) for modeling purposes.  
4. **Saved Preprocessed Dataset:** Ready for Week 3 model training.

---

## Week 3 Milestone Tasks
1. **Feature Engineering:** Created a synthetic label (`flood_risk`) combining Monsoon Intensity, Topography Drainage, and River Management.  
2. **Model Training:** Implemented two models – Logistic Regression and Random Forest Classifier.  
3. **Model Evaluation:**  
   - Accuracy comparison between models  
   - Confusion Matrix  
   - Classification Report (Precision, Recall, F1-Score)  
4. **Insights:**  
   - Random Forest showed higher accuracy compared to Logistic Regression.  
   - Feature importance analysis identified Monsoon Intensity, River Management, and Topography Drainage as critical factors.  

---

## Improvisations Done
- Designed a **synthetic flood risk label** to simulate classification.  
- Added **baseline ML models** (Logistic Regression & Random Forest).  
- Conducted detailed evaluation using accuracy, confusion matrix, and classification report.  
- Visualized **feature importance** for better interpretability of the Random Forest model.  
- Enhanced reproducibility by structuring preprocessing, EDA, and model training in separate notebooks.  

---

## Tools & Libraries
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, KaggleHub  
- **Environment:** Jupyter Notebook / Google Colab  

---

## Folder Structure
