import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

def download_dataset_weather():
  print("Fetching...")
  url = "https://archive-api.open-meteo.com/v1/archive"
  
  # Koordinat Yogyakarta (Pusat)
  params = {
    "latitude": -7.7956,
    "longitude": 110.3695,
    "start_date": "2020-03-01", # 3 Tahun Sebelakang
    "end_date": "2026-03-01",
    "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "surface_pressure"],
    "timezone": "Asia/Bangkok"
  }
  
  responses = openmeteo.weather_api(url, params=params)
  response = responses[0]

  hourly = response.Hourly()
  temp = hourly.Variables(0).ValuesAsNumpy()
  humidity = hourly.Variables(1).ValuesAsNumpy()
  rain = hourly.Variables(2).ValuesAsNumpy()
  pressure = hourly.Variables(3).ValuesAsNumpy()
  
  data = {
    "date": pd.date_range(
      start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
      periods = len(temp),
      freq = pd.Timedelta(seconds = hourly.Interval())
    ),
    "temp": temp,
    "humidity": humidity,
    "rain": rain,
    "pressure": pressure
  }

  df = pd.DataFrame(data)
  
  df['hour'] = df['date'].dt.hour
  df['month'] = df['date'].dt.month

  df.to_csv("weather_dataset.csv", index=False)
  print(f"Success! {len(df)} data saved to weather_dataset.csv")

if __name__ == "__main__":
  download_dataset_weather()