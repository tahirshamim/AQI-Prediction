# ===============================
# train_evaluate_predict.py
# ===============================

import numpy as np
import pandas as pd
import joblib
from pymongo import MongoClient
from datetime import datetime, timezone, UTC, timedelta

from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# -------------------------------
# MongoDB connection
# -------------------------------
client = MongoClient(os.environ["MONGO_URI"])
db = client["aqi_mlops"]

# -------------------------------
# LOAD DATA FROM MONGODB
# -------------------------------
cursor = db.datastore.find().sort("date", 1)
df = pd.DataFrame(list(cursor))

if df.empty:
    raise ValueError("No data found in aqi_features")

df.drop(columns=["_id"], inplace=True, errors="ignore")
df = df.dropna().reset_index(drop=True)

# -------------------------------
# FEATURES & TARGETS
# -------------------------------
FEATURES = [
    'AQI',
    'day',
    'month',
    'dayofweek',
    'aqi_lag_1',
    'aqi_lag_3',
    'aqi_lag_7',
    'aqi_roll_3',
    'aqi_roll_7'
    
]

TARGETS = ["AQI_t+1", "AQI_t+2", "AQI_t+3"]

X = df[FEATURES]
y = df[TARGETS]

# -------------------------------
# TIME-BASED SPLIT
# -------------------------------
split_idx = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# -------------------------------
# SCALING
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# -------------------------------
# MODEL TRAINING
# -------------------------------
model = MultiOutputRegressor(
    ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)
)
model.fit(X_train_scaled, y_train)

# -------------------------------
# EVALUATION
# -------------------------------
y_pred_test = model.predict(X_test_scaled)

evaluation_results = {}

for i, target in enumerate(TARGETS):
    evaluation_results[target] = {
        "MAE":  float(mean_absolute_error(y_test.iloc[:, i], y_pred_test[:, i])),
        "RMSE": float(np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred_test[:, i]))),
        "R2":   float(r2_score(y_test.iloc[:, i], y_pred_test[:, i]))
    }

# -------------------------------
# GET LATEST DATA FOR PREDICTION
# -------------------------------
latest_doc = db.aqi_features.find_one(sort=[("date", -1)])

X_latest = pd.DataFrame(
    [[latest_doc[f] for f in FEATURES]],
    columns=FEATURES
)

X_latest_scaled = scaler.transform(X_latest)

# -------------------------------
# PREDICT (CURRENT MODEL)
# -------------------------------
pred = model.predict(X_latest_scaled)[0]

# -------------------------------
# STORE PREDICTION + EVALUATION
# -------------------------------
prediction_document = {
    "prediction_date": datetime.now(timezone.utc),
    "input_date": latest_doc.get("date"),

    "predictions": {
        "AQI_t+1": float(pred[0]),
        "AQI_t+2": float(pred[1]),
        "AQI_t+3": float(pred[2])
    },

    "evaluation_metrics": {
        "AQI_t+1": evaluation_results["AQI_t+1"],
        "AQI_t+2": evaluation_results["AQI_t+2"],
        "AQI_t+3": evaluation_results["AQI_t+3"]
    }
}

db.aqi_predictions.insert_one(prediction_document)

# -------------------------------
# EXPORT PKL FILES  ✅ ADDED
# -------------------------------
joblib.dump(model, "elasticnet_aqi_multi_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(FEATURES, "features.pkl")
joblib.dump(evaluation_results, "evaluation_metrics.pkl")

print("✅ Model trained, evaluated, predicted, stored in MongoDB, and PKLs exported")
datastore_doc = latest_doc.copy()
datastore_doc.pop("_id", None)  # remove Mongo _id

# Add prediction metadata
datastore_doc["prediction_date"] = datetime.now(UTC) + timedelta(hours=5) #PKT time

# Add predicted AQI values
datastore_doc["AQI_t+1"] = float(pred[0])
datastore_doc["AQI_t+2"] = float(pred[1])
datastore_doc["AQI_t+3"] = float(pred[2])

# -------------------------------
# INSERT INTO NEW COLLECTION: datastore
# -------------------------------
db.datastore.insert_one(datastore_doc)

print("✅ Last AQI feature row + predictions stored in datastore")





