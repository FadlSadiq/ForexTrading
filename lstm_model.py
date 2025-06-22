import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class FXLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def create_sequences(data, lookback=30):
    X, y = [], []
    for i in range(data.shape[1] - lookback):
        X.append(data[:, i:i + lookback].T)
        y.append(data[:, i + lookback])
    return np.array(X), np.array(y)


def train_lstm_model(real_fx_data, lookback=30, epochs=20):
    """
    Trains the LSTM model on FX data and returns the model, scalers, lookback, and training history.
    History dict contains 'loss' and 'accuracy' lists of length `epochs`.
    """
    # Normalize each series using MinMaxScaler fit on entire real_fx_data
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
    model = FXLSTM(input_size=3, hidden_size=64, output_size=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Containers for history
    history = {'loss': [], 'accuracy': []}

    # Train
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_tensor)
        loss = criterion(pred, y_tensor)
        loss.backward()
        optimizer.step()

        # Record training loss
        loss_value = loss.item()
        history['loss'].append(loss_value)

        # Compute directional accuracy: sign of (pred - last_input) vs sign of (y - last_input)
        with torch.no_grad():
            last_input = X_tensor[:, -1, :]  # shape (batch, features)
            pred_diff = (pred - last_input)
            true_diff = (y_tensor - last_input)
            correct = ((pred_diff * true_diff) > 0).float()
            # accuracy per feature, then average
            accuracy = correct.mean().item()
            history['accuracy'].append(accuracy)

        print(f'Epoch {epoch+1}, Loss: {loss_value:.6f}, Accuracy: {accuracy:.4f}')

    return model, scalers, lookback, history
