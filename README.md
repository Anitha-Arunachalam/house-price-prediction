# 🏠 House Price Prediction

A Machine Learning project that predicts house prices using regression techniques, with complete data preprocessing, visualization, and model building pipeline.

## 📌 Project Overview
This project aims to build a predictive model that estimates house prices based on various features such as area, number of bedrooms, and other property attributes.

The workflow includes:

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training and evaluation

## 🛠️ Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## 📊 Workflow
1. Data Collection
2. Data Cleaning 
    - Handling missing values
    - Removing outliers
3. Exploratory Data Analysis (EDA)
    - Distribution plots
    - Correlation heatmap
4. Feature Engineering
5. Model Training
6. Model Evaluation

## 📈 Model
- Algorithm Used: Linear Regression 
- Evaluation Metric: R² Score
- Performance: 0.84

🔍 Key Insights

- Area and number of bedrooms have strong influence on house prices
- Outliers significantly affected model performance
- Data preprocessing improved model accuracy
- Correlation analysis helped in selecting important features

📷 Visualizations
- Correlation Heatmap
- Boxplots (Before & After Outliers)
- Histograms

📁 Project Structure
house_price_prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── load.py
│   ├── preprocess.py
│   ├── model.py
│   └── visualize.py
│
├── main.py
├── README.md
└── .gitignore

▶️ How to Run
git clone https://github.com/Anitha-Arunachalam/house-price-prediction.git
cd house-price-prediction
python main.py

🚀 Future Improvements
- Hyperparameter tuning
- Try advanced models (Random Forest, XGBoost)
- Deploy using Flask or Streamlit
- Add user input interface

💼 Resume Value
- Built an end-to-end machine learning pipeline
- Performed data cleaning, EDA, and feature engineering
- Improved model performance through preprocessing
- Demonstrated strong understanding of regression techniques

⭐ If you found this project useful, consider giving it a star!