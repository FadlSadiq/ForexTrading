import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Define LSTM class
class FXLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# Sequence creator
def create_sequences(data, lookback=30):
    X, y = [], []
    for i in range(data.shape[1] - lookback):
        X.append(data[:, i:i + lookback].T)
        y.append(data[:, i + lookback])
    return np.array(X), np.array(y)

# Wrap the training into a function
def train_lstm_model(real_fx_data, lookback=30, epochs=20):
    # Normalize each series
    scalers = [MinMaxScaler() for _ in range(3)]
    fx_scaled = np.stack([
        scalers[i].fit_transform(real_fx_data[i].reshape(-1, 1)).flatten()
        for i in range(3)
    ])

    # Prepare sequences
    X, y = create_sequences(fx_scaled, lookback)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Model + training setup
    model = FXLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Train
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_tensor)
        loss = criterion(pred, y_tensor)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item():.6f}')

    return model, scalers, lookback  # the last one is optional
