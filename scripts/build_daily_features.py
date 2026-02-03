import pandas as pd
from pymongo import MongoClient
from datetime import datetime, timedelta, timezone,UTC
import os
# ------------------ MongoDB ------------------
client = MongoClient(os.environ["MONGO_URI"])
db = client["aqi_mlops"]

# ------------------ STEP 1: Compute daily AQI from hourly ------------------
end = datetime.now(UTC) + timedelta(hours=5) #PKT time
start = end - timedelta(hours=24)

cursor = db.raw_aqi_hourly.find({
    "timestamp": {"$gte": start, "$lt": end}
})

hourly_df = pd.DataFrame(list(cursor))
if hourly_df.empty:
    raise Exception("No hourly data found")

daily_aqi = float(hourly_df["aqi"].mean())

# date = datetime(end.year, end.month, end.day, tzinfo=timezone.utc)
date = end

# ------------------ STEP 2: Load previous feature history ------------------
history_df = pd.DataFrame(
    list(db.aqi_features.find({}, {"_id": 0}))
).sort_values("date")
if len(history_df) < 8:
    raise Exception("❌ Not enough historical data in aqi_features (need at least 7 rows)")

# ------------------ STEP 3: Append today AQI ------------------
new_row = {
    "date": date,
    "AQI": daily_aqi
}

history_df = pd.concat(
    [history_df, pd.DataFrame([new_row])],
    ignore_index=True
)

# ------------------ STEP 4: Feature engineering ------------------
history_df["date"] = pd.to_datetime(history_df["date"], utc=True)


history_df["day"] = history_df["date"].dt.day
history_df["month"] = history_df["date"].dt.month
history_df["dayofweek"] = history_df["date"].dt.dayofweek

history_df["aqi_lag_1"] = history_df["AQI"].shift(1)
history_df["aqi_lag_3"] = history_df["AQI"].shift(3)
history_df["aqi_lag_7"] = history_df["AQI"].shift(7)

history_df["aqi_roll_3"] = history_df["AQI"].shift(1).rolling(3).mean()
history_df["aqi_roll_7"] = history_df["AQI"].shift(1).rolling(7).mean()

latest = history_df.dropna().iloc[-1].to_dict()

# ------------------ STEP 5: Store ONLY in aqi_features ------------------
db.aqi_features.update_one(
    {"date": latest["date"]},
    {"$set": latest},
    upsert=True
)

print("✅ Daily AQI + features stored in aqi_features")




