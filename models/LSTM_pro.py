import torch
import torch.nn as nn

from models.extractor import TimesNetC
from models.Att import CWSA
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
    def __init__(self, input_size, hidden_size=64, num_layers=3, pred_len=90):
        super().__init__()

        self.lstm_block = DeepLSTMBlock(input_size, hidden_size, num_layers)
        # 使用 TimesNetC 作为特征提取器
        self.feature = TimesNetC(in_len=input_size,out_len=pred_len,channels=hidden_size)
        # 使用 CWSA 作为频域特征选择
        self.cwsa = CWSA(feature_dim=hidden_size)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, pred_len)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out = self.lstm_block(x)
        out = self.feature(out)
          # (batch, seq_len, hidden)
        out = self.cwsa(out)  # 频域特征选择
        # out: (batch, seq_len, hidden)
        out = out[:, -1, :]  # (batch, hidden)
        last_hidden = out  # (batch, hidden)
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
# ...existing code...

if __name__ == "__main__":
    # 假设输入序列长度为30，特征维度为10
    batch_size = 4
    seq_len = 90
    input_dim = 10
    pred_len = 90

    # 随机生成输入数据
    x = torch.randn(batch_size, seq_len, input_dim)

    # 实例化模型
    model = LSTM(input_size=input_dim, hidden_size=64, num_layers=3, pred_len=pred_len)

    # 前向传播
    output = model(x)

    print("输出形状:", output.shape)  # 期望形状: (batch_size, seq_len, pred_len) 或 (batch_size, pred_len)