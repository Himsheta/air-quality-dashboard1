# 🌍 Air Quality Prediction Dashboard

**Data Science + Machine Learning** project that predicts Air Quality Index (AQI) using environmental data, featuring EDA, model training, and an interactive **Streamlit dashboard**.

---

## 🧠 Project Overview

This project focuses on analyzing air quality data and predicting AQI values based on pollutants like PM2.5, PM10, NO, CO, and SO2.  
It demonstrates the **full ML lifecycle** — from data cleaning to visualization, model training, and deployment.

### ✨ Features
- ✅ Data cleaning & processing using **Pandas**
- 📊 EDA using **Seaborn** & **Matplotlib**
- 🧠 AQI prediction using **RandomForestRegressor**
- 🖥 Interactive dashboard built with **Streamlit**
- ☁️ Easy deployment on **Streamlit Cloud**

---

## 🗂 Project Structure

air-quality-dashboard/
│── data/
│ ├── raw/ # Original dataset (from Kaggle)
│ └── processed/ # Cleaned dataset
│
│── src/
│ ├── clean_data.py # Data preprocessing
│ ├── train_model.py # ML model training
│ └── dashboard.py # Streamlit dashboard
│
│── notebooks/ # (Optional) EDA notebooks
│── requirements.txt
│── .gitignore
│── README.md
