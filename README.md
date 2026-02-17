#  AQI Forecasting MLOps Pipeline

An end-to-end Machine Learning Operations (MLOps) project that collects real-time Air Quality Index (AQI) data, engineers predictive features, trains forecasting models, and serves multi-day AQI predictions through an interactive dashboard.

This project was developed as part of an internship to demonstrate practical skills in data engineering, machine learning, and automated ML pipelines.

---

##  Project Overview

This system builds a production-style pipeline for AQI forecasting using real-time environmental data. It integrates automated data ingestion, feature engineering, model training, evaluation, and prediction storage.

The system predicts AQI for the next **3 days** using historical trends and engineered time-series features.

---

##  System Architecture

The pipeline consists of the following components:

1. **Hourly AQI Data Pipeline**

   * Fetches real-time AQI data from external API
   * Stores raw hourly data in MongoDB

2. **Feature Engineering Pipeline**

   * Aggregates hourly data into daily AQI
   * Creates lag and rolling statistical features
   * Stores processed features in a feature datastore

3. **Training Pipeline**

   * Uses historical engineered features
   * Trains a multi-output regression model
   * Evaluates performance with standard metrics

4. **Prediction Pipeline**

   * Generates 3-day AQI forecasts
   * Stores predictions and evaluation metrics

5. **Interactive Dashboard**

   * Displays predictions and trends
   * Provides real-time monitoring

---

##  Pipeline Workflow

API → Raw Data Storage → Feature Engineering → Model Training → Prediction → Dashboard

The pipeline is designed to simulate real-world MLOps workflows with modular components.

---

##  Data Source

AQI data is collected from a public AQI API and trained using approximately **one year of historical data**.

This provides enough seasonal and temporal variation for meaningful forecasting.

---

##  Feature Engineering

The following features are engineered to capture temporal patterns:

* Current AQI
* Day, month, day of week
* Lag features (1, 3, 7 days)
* Rolling averages (3 and 7 days)

These features help the model learn trends, seasonality, and short-term fluctuations.

---

##  Model Selection

Multiple models were evaluated:

* Random Forest
* Gradient Boosting
* Ridge Regression
* Lasso Regression
* **Elastic Net (selected)**

### Why Elastic Net?

Elastic Net combines L1 and L2 regularization, which:

* Reduces overfitting
* Handles correlated features well
* Provides stable predictions
* Performs better on small-to-medium datasets

It achieved the best balance of accuracy and generalization in experiments.

---

##  Evaluation Metrics

The model is evaluated using:

* **MAE (Mean Absolute Error)**
* **RMSE (Root Mean Square Error)**
* **R² Score**

These metrics ensure reliable forecasting performance.

---

##  Database Design

MongoDB is used for flexible and scalable storage:

* `raw_aqi_hourly` → raw hourly data
* `aqi_features` → engineered features
* `datastore` → training dataset
* `aqi_predictions` → stored predictions

This structure supports reproducibility and pipeline automation.

---

##  Installation & Setup

### 1. Clone Repository

```
git clone https://github.com/tahirshamim/AQI-Prediction.git
cd AQI-Prediction
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Configure Environment

Set your MongoDB connection string and API token in environment variables.

### 4. Run Pipelines

```
python fetch_hourly_data.py
python feature_pipeline.py
python train_evaluate_predict.py
```

### 5. Launch Dashboard

```
streamlit run app.py
```

---

## 🔧 Technologies Used

* Python
* Pandas & NumPy
* Scikit-learn
* MongoDB
* Streamlit
* Requests API

---

##  Key Learning Outcomes

This project demonstrates:

* End-to-end MLOps workflow design
* Time-series feature engineering
* Multi-step forecasting
* Model experimentation and evaluation
* Automated data pipelines
* Database integration
* Dashboard development

---

##  Future Improvements

* Integration with Airflow for scheduling
* Cloud feature store deployment
* Deep learning forecasting models
* SHAP explainability integration
* Real-time alert system
* Multi-city AQI forecasting

---

##  Author

Developed as part of an internship project focused on practical MLOps and predictive analytics.

---

##  License

This project is for educational and demonstration purposes.

