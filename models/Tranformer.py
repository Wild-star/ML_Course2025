import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(1), :]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, pred_len, dim_feedforward=1024, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, pred_len)  # 修改这里
        self.pred_len = pred_len
        self.d_model = d_model
        
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def forward(self, src):
        src = self.input_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src_mask = self.generate_square_subsequent_mask(src.size(1)).to(src.device)
        output = self.transformer_encoder(src, src_mask)
        output = output[:, -1, :]  # 取最后一个时间步
        output = self.decoder(output)  # 输出 [batch, pred_len]
        return output
    
def save_model(model, path="lstm_model.pt"):
    torch.save(model.state_dict(), path)
def load_model(path, input_dim, d_model, nhead, num_layers, pred_len, dim_feedforward=1024, dropout=0.1, device='cpu'):
    model = TransformerModel(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        pred_len=pred_len,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
