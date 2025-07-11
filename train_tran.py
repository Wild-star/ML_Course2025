import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from models.Tranformer import TransformerModel
from sklearn.preprocessing import StandardScaler
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len, 0])  # 假设第一列是目标变量
    return np.array(X), np.array(y)

def calculate_metrics(y_true, y_pred):
    """计算MSE和MAE指标"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, mae

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, patience=1000):
    best_val_loss = float('inf')
    best_model = None
    writer = SummaryWriter(log_dir='runs/transformer90')
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        all_train_preds = []
        all_train_labels = []
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            
            optimizer.step()
            train_loss += loss.item()
            
            # 收集预测和标签用于计算指标
            all_train_preds.extend(outputs.detach().cpu().numpy())
            all_train_labels.extend(y_batch.cpu().numpy())
        
        # 计算训练集指标
        train_mse, train_mae = calculate_metrics(all_train_labels, all_train_preds)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                # 收集预测和标签用于计算指标
                all_val_preds.extend(outputs.cpu().numpy())
                all_val_labels.extend(y_batch.cpu().numpy())
        
        # 计算验证集指标
        val_mse, val_mae = calculate_metrics(all_val_labels, all_val_preds)
        
        # 学习率调整
        #scheduler.step(val_loss)
        
        # 打印训练信息
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}/{epochs}')
            print(f'Train Loss: {avg_train_loss:.6f}, MSE: {train_mse:.6f}, MAE: {train_mae:.6f}')
            print(f'Val Loss: {avg_val_loss:.6f}, MSE: {val_mse:.6f}, MAE: {val_mae:.6f}')
            save_dir = 'transformer90'
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, f'transformer_epoch{epoch+1}.pth'))
            
        # === TensorBoard记录 ===
       
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('MSE/train', train_mse, epoch)
        writer.add_scalar('MSE/val', val_mse, epoch)
        writer.add_scalar('MAE/train', train_mae, epoch)
        writer.add_scalar('MAE/val', val_mae, epoch)
        # 早停策略
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict().copy()
        
    
    # 加载最佳模型
    torch.save(best_model, os.path.join(save_dir,f'transformer_forecast_model.pth'))
    model.load_state_dict(best_model)

    return model, best_val_loss

def main():
    parser = argparse.ArgumentParser(description='Transformer Time Series Forecasting')
    parser.add_argument('--data_path', type=str, default='./file/train_final.csv', help='Path to the data file')
    parser.add_argument('--seq_len', type=int, default=90, help='Sequence length')
    parser.add_argument('--pred_len', type=int, default=90, help='Prediction length')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs')
    parser.add_argument('--d_model', type=int, default=128, help='Transformer model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=512, help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=1000, help='Early stopping patience')
    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
# 数据加载与处理
    df = pd.read_csv(args.data_path)
    df = df.drop(columns=["date"])
    features = df.drop(columns=["Global_active_power"]).values
    target = df["Global_active_power"].values.reshape(-1, 1)

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    features = feature_scaler.fit_transform(features)
    target = target_scaler.fit_transform(target)
    def create_sequences(features, target, seq_len, pred_len):
        X, y = [], []
        for i in range(len(features) - seq_len - pred_len):
            X.append(features[i:i+seq_len])
            y.append(target[i+seq_len:i+seq_len+pred_len].flatten())
        return np.array(X), np.array(y)

    X, y = create_sequences(features, target, args.seq_len, args.pred_len)
    # 划分训练集和验证集
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # 创建数据加载器
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # 初始化模型
    input_dim = X.shape[2]  # 特征维度
    model = TransformerModel(
        input_dim=input_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        pred_len=args.pred_len,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    ).to(device)
    # 输出模型参数量
    print(f'Total parameters: {count_parameters(model)}')
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=15, verbose=True
    )
    
    # 训练模型
    best_model, best_loss = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        device, args.epochs, args.patience
    )
    
    # 保存模型
    
    
    print(f'Model saved with best validation loss: {best_loss:.6f}')
    # === 计算并输出最终验证集的MSE和MAE ===
    model.eval()
    all_val_preds = []
    all_val_labels = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            all_val_preds.extend(outputs.cpu().numpy())
            all_val_labels.extend(y_batch.cpu().numpy())
    final_mse, final_mae = calculate_metrics(all_val_labels, all_val_preds)
    print(f'Final Validation MSE: {final_mse:.6f}, MAE: {final_mae:.6f}')

if __name__ == '__main__':
    main()    