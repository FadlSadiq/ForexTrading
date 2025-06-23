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