# fx_cnn_bigru_attn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FXCNNBiGRUAttn(nn.Module):
    def __init__(self, input_size=3, cnn_channels=64, gru_hidden=64, output_size=3):
        super().__init__()
        self.cnn = nn.Conv1d(in_channels=input_size, out_channels=cnn_channels, kernel_size=3, padding=1)
        self.bigru = nn.GRU(input_size=cnn_channels, hidden_size=gru_hidden, batch_first=True, bidirectional=True)
        self.attn_fc = nn.Linear(gru_hidden * 2, 1)
        self.out_fc = nn.Linear(gru_hidden * 2, output_size)

    def attention(self, x):
        attn_weights = torch.softmax(self.attn_fc(x), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(attn_weights * x, dim=1)          # (batch, hidden*2)
        return context

    def forward(self, x):
        x = x.permute(0, 2, 1)                  # (batch, features, seq_len)
        x = F.relu(self.cnn(x))                 # (batch, cnn_channels, seq_len)
        x = x.permute(0, 2, 1)                  # (batch, seq_len, cnn_channels)
        out, _ = self.bigru(x)                  # (batch, seq_len, hidden*2)
        context = self.attention(out)           # (batch, hidden*2)
        return self.out_fc(context)             # (batch, output_size)