# 频域特征选择（FFT实现）
import torch
import torch.nn as nn

class CWSA(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.spectral_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, feature_dim),
            nn.Sigmoid()  # 生成[0,1]频域掩码
        )

    def forward(self, x):
        x_fft = torch.fft.rfft(x, dim=1)  # 实部FFT
        mag = torch.abs(x_fft) 
        mask = self.spectral_gate(mag.mean(dim=1)) 
        return x * mask.unsqueeze(1)  # 频域滤波