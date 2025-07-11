import torch
import torch.nn as nn


class DeepLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.dropout(output)
        return output


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, pred_len=90):
        super().__init__()
        self.lstm_block = DeepLSTMBlock(input_size, hidden_size, num_layers)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, pred_len)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out = self.lstm_block(x)  # (batch, seq_len, hidden)
        last_hidden = out[:, -1, :]  # (batch, hidden)
        prediction = self.output_layer(last_hidden)  # (batch, pred_len)
        return prediction


# 模型保存函数

def save_model(model, path="lstm_model.pt"):
    torch.save(model.state_dict(), path)


# 模型加载函数

def load_model(path, input_size, hidden_size, num_layers, pred_len):
    model = LSTM(input_size, hidden_size, num_layers, pred_len)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
