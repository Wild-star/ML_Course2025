
# ============================

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from models.LSTM import LSTM, save_model, load_model
from torch.utils.tensorboard import SummaryWriter

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parse_args():
    parser = argparse.ArgumentParser(description="LSTM 训练脚本")
    parser.add_argument('--seq_len', type=int, default=90, help='输入时间窗口')
    parser.add_argument('--pred_len', type=int, default=90, help='预测未来天数')
    parser.add_argument('--epochs', type=int, default=10000, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128, help='批大小')
    parser.add_argument('--hidden_size', type=int, default=64, help='LSTM隐藏层大小')
    parser.add_argument('--num_layers', type=int, default=4, help='LSTM层数')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='学习率')
    parser.add_argument('--n_runs', type=int, default=5, help='重复训练次数')
    parser.add_argument('--data_path', type=str, default="./file/train_final.csv", help='数据文件路径')
    return parser.parse_args()

def main():
    args = parse_args()
    seq_len = args.seq_len
    pred_len = args.pred_len
    num_epochs = args.epochs
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    learning_rate = args.learning_rate
    n_runs = args.n_runs
    data_path = args.data_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据加载与处理
    df = pd.read_csv(data_path)
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

    X, y = create_sequences(features, target, seq_len, pred_len)

    mse_list, mae_list = [], []

    for run in range(n_runs):
        print(f"\n==== Run {run+1} ====")
        model_name = f"LSTM_run{run+1}_pred{pred_len}"
        writer = SummaryWriter(log_dir=f"runs/{model_name}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=run)
        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = LSTM(input_size=X.shape[2], hidden_size=hidden_size, num_layers=num_layers, pred_len=pred_len).to(device)
        print(f"模型可训练参数量: {count_parameters(model)}")
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

        # 训练
        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            writer.add_scalar("Loss/train", avg_loss, epoch)
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")

        # 保存模型
        save_model(model, path=f"LSTM_pred{pred_len}/LSTM_run{run+1}_pred{pred_len}.pt")

        # 评估
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                out = model(xb).cpu()
                preds.append(out.numpy())
                trues.append(yb.numpy())
        preds = target_scaler.inverse_transform(np.vstack(preds))
        trues = target_scaler.inverse_transform(np.vstack(trues))

        mse = np.mean((preds - trues)**2)
        mae = np.mean(np.abs(preds - trues))
        mse_list.append(mse)
        mae_list.append(mae)

        writer.add_scalar("Metrics/MSE", mse, 0)
        writer.add_scalar("Metrics/MAE", mae, 0)
        writer.close()

        print(f"Run {run+1} - MSE: {mse:.4f}, MAE: {mae:.4f}")

    mse_mean, mse_std = np.mean(mse_list), np.std(mse_list)
    mae_mean, mae_std = np.mean(mae_list), np.std(mae_list)
    print("\n==== Final Result ({} Runs) ====".format(n_runs))
    print(f"Avg MSE: {mse_mean:.4f} ± {mse_std:.4f}")
    print(f"Avg MAE: {mae_mean:.4f} ± {mae_std:.4f}")

if __name__ == "__main__":
    main()
