import torch
import numpy as np
import pandas as pd
import joblib
import openmeteo_requests
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

INFLUX_URL = "http://localhost:8086" # Change With Your InfluxDB
INFLUX_TOKEN = "xxxxxxxxxx" # Change With Your Token
INFLUX_ORG = "yyyyyyyyy" # Change Your Org
INFLUX_BUCKET = "weather_forecast" # Change Your Bucket

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Running on: {device}")

scaler = joblib.load('../model-train/models/scaler.pkl')

hidden_size = 128
hidden_layer = 3

class WeatherLSTM(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.lstm = torch.nn.LSTM(input_size=6, hidden_size=hidden_size, num_layers=hidden_layer, batch_first=True)
    self.linear = torch.nn.Linear(hidden_size, 4)
  def forward(self, x):
    out, _ = self.lstm(x)
    return self.linear(out[:, -1, :])

model = WeatherLSTM().to(device)
model.load_state_dict(torch.load('../model-train/models/weather_model.pt', map_location=device))
model.eval()

def fetch_real_history():
  print("🛰️ Fetching real history from API...")
  openmeteo = openmeteo_requests.Client()
  url = "https://api.open-meteo.com/v1/forecast"
  params = {
    "latitude": -7.7956, "longitude": 110.3695,
    "past_days": 1, "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "surface_pressure"]
  }
  responses = openmeteo.weather_api(url, params=params)
  hourly = responses[0].Hourly()
  
  now = datetime.now()
  data = []
  for i in range(24):
    idx = (len(hourly.Variables(0).ValuesAsNumpy()) - 24) + i
    t_hash = (now - timedelta(hours=23-i))
    data.append([
      hourly.Variables(0).ValuesAsNumpy()[idx],
      hourly.Variables(1).ValuesAsNumpy()[idx],
      hourly.Variables(2).ValuesAsNumpy()[idx],
      hourly.Variables(3).ValuesAsNumpy()[idx],
      t_hash.hour,
      t_hash.month
    ])
  
  return pd.DataFrame(data, columns=['temp', 'humidity', 'rain', 'pressure', 'hour', 'month'])

def forecast_to_grafana(days=90):
  df_history = fetch_real_history()
  features = ['temp', 'humidity', 'rain', 'pressure', 'hour', 'month']
  
  current_batch_np = scaler.transform(df_history[features].values)
  current_batch = torch.tensor(current_batch_np, dtype=torch.float32).unsqueeze(0).to(device)
  
  client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG, timeout=60000)
  write_api = client.write_api(write_options=SYNCHRONOUS)

  print(f"🚀 Forecasting {days} days and pushing to Grafana...")
  
  start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
  points = []
  
  with torch.no_grad():
    for i in range(days * 24):
      pred = model(current_batch) 
      pred_np = pred.cpu().numpy() 
      
      pred_time = start_time + timedelta(hours=i+1)
      
      dummy = np.zeros((1, 6))
      dummy[:, 0:4] = pred_np
      dummy[:, 4] = pred_time.hour
      dummy[:, 5] = pred_time.month
      
      unscaled = scaler.inverse_transform(dummy)[0]
      t, h, r, p = unscaled[0], unscaled[1], max(0, unscaled[2]), unscaled[3]

      point = Point("forecast") \
        .tag("location", "yogyakarta") \
        .field("temperature", float(t)) \
        .field("humidity", float(h)) \
        .field("rain", float(r)) \
        .field("pressure", float(p)) \
        .time(pred_time, WritePrecision.NS)
      
      points.append(point)
      next_raw_row = np.array([[t, h, r, p, pred_time.hour, pred_time.month]])
      next_scaled_row = scaler.transform(next_raw_row)
      
      new_row_tensor = torch.tensor(next_scaled_row, dtype=torch.float32).unsqueeze(0).to(device)
      current_batch = torch.cat((current_batch[:, 1:, :], new_row_tensor), dim=1)
  print(f"📡 Writing {len(points)} points to InfluxDB...")
  for j in range(0, len(points), 500):
    batch = points[j:j+500]
    write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=batch)
  
  client.close()
  print("✅ Done! Check your Grafana. (Make sure Time Range is now to now+90d)")

if __name__ == "__main__":
  forecast_to_grafana(90)