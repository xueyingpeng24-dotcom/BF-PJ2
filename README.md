Business Forecasting for Multi-Store Retail Sales

This project develops an end-to-end forecasting framework for a large European retail chain with over 600 stores. The objective is to generate accurate 6-week sales forecasts that support inventory planning, staffing decisions, and short-term financial planning while maintaining a practical and scalable forecasting pipeline. The project follows the requirements of the TUM Business Forecasting course project, which emphasizes explainable forecasting methods and business-oriented decision making.

Project Highlights
Performed comprehensive data cleaning and feature engineering on daily sales data from more than 600 stores.
Explored sales patterns, seasonality, promotions, holidays, and store characteristics through exploratory data analysis.
Implemented and compared multiple benchmark forecasting models:

Developed a Global Random Forest model using engineered time-series features and store metadata.
Evaluated model performance using MAPE and MAE, and compared forecasting accuracy across individual stores and store types.
Selected forecasting methods by balancing forecast accuracy, computational efficiency, interpretability, and operational maintainability.
Results

Among all benchmark methods, ETS achieved the best overall forecasting performance, providing the lowest average forecasting error and becoming the preferred statistical benchmark for the retail forecasting task. The Global Random Forest model was further developed to leverage cross-store information and engineered features for scalable forecasting.

Repository Structure
├── notebooks/
│   ├── EDA & Data Cleaning
│   ├── Feature Engineering
│   ├── Benchmark Models
│   └── Global Random Forest
├── data/
│   ├── raw/
│   └── processed/
├── results/
├── poster/
└── README.md

Technologies
Python
Pandas
NumPy
Scikit-learn
StatsForecast
pmdarima
Matplotlib
Learning Outcomes

This project demonstrates practical applications of:

Time Series Forecasting
Feature Engineering
Statistical Forecasting Models
Global Machine Learning Models
Forecast Accuracy Evaluation
Business oriented Model Selection
Retail Demand Forecasting
