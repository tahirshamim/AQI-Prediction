from pymongo import MongoClient
import requests
from datetime import datetime, UTC

client = MongoClient("mongodb+srv://tahirbinshamim_db_user:42034700@cluster0.2gfzzp7.mongodb.net/?appName=Cluster0")
db = client["aqi_mlops"]

API_URL = "https://api.waqi.info/feed/A545395/?token=c417583af21dac5a62ca39cfc874cc80162e475a"

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
        "timestamp": datetime.now(UTC),
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
