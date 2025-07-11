import torch
import torch.nn as nn
import torch.nn.functional as F

from models.emb_layers import DataEmbedding
from models.conv_layers import Inception_Block_V1


def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, in_len, out_len, d_model=32, d_ff=64):
        super(TimesBlock, self).__init__()
        self.seq_len = in_len
        self.pred_len = out_len
        self.k = 5
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=6),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=6)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)
        res = []
        for i in range(self.k):
            period = period_list[i]
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        res = res + x
        return res  # [B, in_len+out_len, C]


class TimesNetC(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """
    def __init__(self, in_len=None, out_len=None,channels=None,freq='d',n_layers = 3,d_ff=64,d_model = 32,dropout=0.1):
        super(TimesNetC, self).__init__()
        self.seq_len =in_len
        self.pred_len = out_len
        self.channels = channels
        self.d_ff = d_ff
        self.d_model = d_model
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.ModuleList([TimesBlock(in_len=in_len,out_len=out_len,d_ff=d_ff,d_model=d_model) for _ in range(n_layers)])
        self.enc_embedding = DataEmbedding(channels, d_model=self.d_model, freq=freq, dropout=dropout)
        self.layer = n_layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, channels, bias=True)

    def forward(self, x):
        # 归一化
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x/stdev

        # 补齐长度到 in_len + out_len
        total_len = self.seq_len + self.pred_len
        if x.shape[1] < total_len:
            pad_len = total_len - x.shape[1]
            x = torch.cat([x, torch.zeros(x.shape[0], pad_len, x.shape[2]).to(x.device)], dim=1)
        elif x.shape[1] > total_len:
            x = x[:, :total_len, :]

        # 嵌入
        enc_out = self.enc_embedding(x, x_mark=None)  # [B, in_len+out_len, d_model]
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        dec_out = self.projection(enc_out)  # [B, in_len+out_len, C]

        # 反归一化
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, total_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, total_len, 1))
        return dec_out[:, -self.pred_len:, :]


if __name__ == "__main__":
    from types import SimpleNamespace

    args = SimpleNamespace(
        in_len=36,
        out_len=24,
        data_dim=8,
        d_model=32,
        d_ff=64,
        n_layers=2,
        freq='h',
        dropout=0.1,
        cuda=0
    )

    model = TimesNetC(in_len=args.in_len, out_len=args.out_len, channels=args.data_dim, freq=args.freq, d_ff=args.d_ff, d_model=args.d_model, dropout=args.dropout)
    x = torch.randn(4, args.in_len, args.data_dim)  # [batch, in_len, data_dim]
    output = model(x)
    print("输入形状:", x.shape)
    print("输出形状:", output.shape)
