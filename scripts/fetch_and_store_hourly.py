from pymongo import MongoClient
import requests
from datetime import datetime, UTC
import os

client = MongoClient(os.environ["MONGO_URI"])
db = client["aqi_mlops"]

API_URL = os.environ["WAQI_API_KEY"]

try:
    res = requests.get(API_URL, timeout=10)
    res.raise_for_status()
    data = res.json()

    if data.get("status") != "ok":
        raise ValueError("API returned non-ok status")

    aqi_raw = data.get("data", {}).get("aqi")

    if aqi_raw in (None, "-", ""):
        raise ValueError("AQI value not found")

    aqi = int(aqi_raw)

    doc = {
        "timestamp": datetime.now(UTC) + timedelta(hours=5), #PKT time
        "aqi": aqi
    }

    db.raw_aqi_hourly.insert_one(doc)
    print("Hourly AQI stored:", aqi)

except requests.exceptions.RequestException as e:
    print("Network/API error:", e)

except ValueError as e:
    print("Data error:", e)

except Exception as e:
    print("Unexpected error:", e)


