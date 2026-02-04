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


latest_doc = db.aqi_features.find_one(sort=[("date", -1)])
# Get last 4 rows in ascending order (oldest → newest)
docs = list(
    db.datastore.find()
    .sort("date", -1)
    .limit(3)
)
docs.reverse()          
if len(docs) < 3:
    raise ValueError("Need at least 4 rows in datastore")

# Assign for readability
fourth_last = docs[-3]   # date = t-3
third_last  = docs[-2]   # date = t-2
second_last = docs[-1]   # date = t-1


latest_aqi = latest_doc['AQI']

# -------------------------------
# Update target columns
# -------------------------------

# 2nd last row → AQI_t+1
db.datastore.update_one(
    {"_id": second_last["_id"]},
    {"$set": {"AQI_t+1": latest_aqi}}
)

# 3rd last row → AQI_t+2
db.datastore.update_one(
    {"_id": third_last["_id"]},
    {"$set": {"AQI_t+2": latest_aqi}}
)

# 4th last row → AQI_t+3
db.datastore.update_one(
    {"_id": fourth_last["_id"]},
    {"$set": {"AQI_t+3": latest_aqi}}
)



datastore_doc = latest_doc.copy()
datastore_doc.pop("_id", None)  # remove Mongo _id
datastore_doc["AQI_t+1"] = np.nan
datastore_doc["AQI_t+2"] = np.nan
datastore_doc["AQI_t+3"] = np.nan
db.datastore.insert_one(datastore_doc)
print("✅ Backfilled AQI_t+1, t+2, t+3 using real AQI")

# -------------------------------
# LOAD DATA FROM MONGODB
# -------------------------------
cursor = db.datastore.find().sort("date", 1)
df = pd.DataFrame(list(cursor))

if df.empty:
    raise ValueError("No data found in aqi_features")

df.drop(columns=["_id"], inplace=True, errors="ignore")
df = df.dropna().reset_index(drop=True)
REQUIRED_ROWS = 8  # max lag (7) + 1

if len(df) < REQUIRED_ROWS:
    raise Exception(
        f"Not enough rows to compute lag/rolling features. "
        f"Found {len(df)}, need at least {REQUIRED_ROWS}"
    )

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
    "prediction_date": datetime.now(UTC) + timedelta(hours=5),
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

print("Prediction Inserted")
# -------------------------------
# EXPORT PKL FILES  ✅ ADDED
# -------------------------------
# joblib.dump(model, "elasticnet_aqi_multi_model.pkl")
# joblib.dump(scaler, "scaler.pkl")
# joblib.dump(FEATURES, "features.pkl")
# joblib.dump(evaluation_results, "evaluation_metrics.pkl")
# print("✅ Model trained, evaluated, predicted, stored in MongoDB, and PKLs exported")
