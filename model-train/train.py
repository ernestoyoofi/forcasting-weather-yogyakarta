import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import joblib
from torch.utils.data import DataLoader, TensorDataset

# First on time, create model folders
if not os.path.exists("models"):
  os.makedirs("models")

# 1. Load Data
print("1. Load dataset")
df = pd.read_csv("../dataset/weather_dataset.csv")
features = ['temp', 'humidity', 'rain', 'pressure', 'hour', 'month']
data = df[features].values

# 2. Normalization
print("2. Normalization")
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
joblib.dump(scaler, 'models/scaler.pkl') # Save scaler for production

# 3. Create Sequence
print("3. Create Sequence")
def create_sequences(data, seq_length):
  x, y = [], []
  for i in range(len(data) - seq_length):
    x.append(data[i:i+seq_length])
    y.append(data[i+seq_length, 0:4])
  return np.array(x), np.array(y)

X, y = create_sequences(scaled_data, 24)
X = torch.tensor(X, dtype=torch.float32).cpu()
y = torch.tensor(y, dtype=torch.float32).cpu()

#4. Model Architecture (Simple LSTM)
hidden_size=128
# hidden_size=256
hidden_layer=3
print("4. Model Architecture")
class WeatherLSTM(nn.Module):
  def __init__(self):
    super().__init__()
    self.lstm = nn.LSTM(input_size=6, hidden_size=hidden_size, num_layers=hidden_layer, batch_first=True)
    self.linear = nn.Linear(hidden_size, 4)
    
  def forward(self, x):
    out, _ = self.lstm(x)
    return self.linear(out[:, -1, :])

train_data = TensorDataset(X, y)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Accelerator: {device}")
model = WeatherLSTM().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. Training Loop
print("5. Training Loop")
# epoch_loop = 500 # 50 Round
epoch_loop = 300 # 50 Round
for epoch in range(epoch_loop):
  model.train()
  epoch_loss = 0

  for batch_X, batch_y in train_loader:
    batch_X, batch_y = batch_X.to(device), batch_y.to(device)

    output = model(batch_X)
    loss = criterion(output, batch_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    epoch_loss += loss.item()

  if (epoch + 1) % 10 == 0:
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
  else:
    print(f"Epoch {epoch+1}/{epoch_loop}")


  # print(f"Epoch {epoch+1}/{epoch_loop}")
  # model.train()
  # optimizer.zero_grad()
  # output = model(X)
  # loss = criterion(output, y)
  # loss.backward()
  # optimizer.step()
  # if (epoch+1) % 10 == 0:
  #   print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("6. Save Model AI")
torch.save(model.state_dict(), "models/weather_model.pt")
print("The model was successfully built in models/weather_model.pt")