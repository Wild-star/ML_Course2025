# evaluate_plot_lstm_v2.py
# ============================
# ============================

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models.LSTM import load_model
import os
from torch.utils.data import Subset
# ----------------------------
# 参数配置
# ----------------------------
n_runs = 5  # 子集数量
seq_len = 90
pred_len = 365
batch_size = 256
hidden_size = 64
num_layers = 4
n_runs = 5
model_prefix = "./LSTM_pred365/LSTM_run"
model_ext = "_pred365.pt"
# ----------------------------
# 加载与预处理数据
# ----------------------------
df = pd.read_csv("./file/test_final.csv")
df = df.drop(columns=["date"])

features = df.drop(columns=["Global_active_power"]).values
target = df["Global_active_power"].values.reshape(-1, 1)

feature_scaler = StandardScaler()
target_scaler = StandardScaler()
features = feature_scaler.fit_transform(features)
target = target_scaler.fit_transform(target)

# 构造序列
def create_sequences(features, target, seq_len, pred_len):
    X, y = [], []
    for i in range(len(features) - seq_len - pred_len):
        X.append(features[i:i+seq_len])
        y.append(target[i+seq_len:i+seq_len+pred_len].flatten())
    return np.array(X), np.array(y)

X, y = create_sequences(features, target, seq_len, pred_len)
X_tensor = torch.Tensor(X)
y_tensor = torch.Tensor(y)
dataset = TensorDataset(X_tensor, y_tensor)
total_len = len(dataset)
indices = np.arange(total_len)
np.random.seed(42)  # 保证可复现
np.random.shuffle(indices)
split_size = total_len // n_runs
splits = [indices[i*split_size:(i+1)*split_size] for i in range(n_runs-1)]
splits.append(indices[(n_runs-1)*split_size:])  # 最后一份包含剩余全部

data_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=False)

# ----------------------------
# 多轮模型评估
# ----------------------------
mse_list, mae_list = [], []
preds_all, trues_all = None, None
model_path = f"{model_prefix}2{model_ext}"
model = load_model(
    path=model_path,
    input_size=X.shape[2],
    hidden_size=hidden_size,
    num_layers=num_layers,
    pred_len=pred_len
)
model.eval()
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {count_parameters(model)}")
for run, split_idx in enumerate(splits):
    subset = Subset(dataset, split_idx)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb)
            preds.append(out.numpy())
            trues.append(yb.numpy())
    preds = target_scaler.inverse_transform(np.vstack(preds))
    trues = target_scaler.inverse_transform(np.vstack(trues))
    if preds_all is None:
        preds_all, trues_all = preds, trues
    mse = mean_squared_error(trues, preds)
    mae = mean_absolute_error(trues, preds)
    mse_list.append(mse)
    mae_list.append(mae)
    print(f"Subset {run+1}: MSE = {mse:.4f}, MAE = {mae:.4f}")

# ----------------------------
# 输出统计信息
# ----------------------------
print("\n==== Final Result (Multi-Runs) ====")
print(f"Avg MSE: {np.mean(mse_list):.4f} ± {np.std(mse_list):.4f}")
print(f"Avg MAE: {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")
for run in range(n_runs):
    model_path = f"{model_prefix}{run+1}{model_ext}"
    if not os.path.exists(model_path):
        print(f"模型 {model_path} 不存在，跳过...")
        continue

    model = load_model(
        path=model_path,
        input_size=X.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
        pred_len=pred_len
    )
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in data_loader:
            out = model(xb)
            preds.append(out.numpy())
            trues.append(yb.numpy())

    preds = target_scaler.inverse_transform(np.vstack(preds))
    trues = target_scaler.inverse_transform(np.vstack(trues))

    if preds_all is None:
        preds_all, trues_all = preds, trues
    
    mse = np.mean((preds - trues) ** 2)
    mae = np.mean(np.abs(preds - trues))
    mse_list.append(mse)
    mae_list.append(mae)

    print(f"Run {run+1}: MSE = {mse:.4f}, MAE = {mae:.4f}")

# ----------------------------
# 输出统计信息
# ----------------------------
print("\n==== Final Result (Multi-Runs) ====")
print(f"Avg MSE: {np.mean(mse_list):.4f} ± {np.std(mse_list):.4f}")
print(f"Avg MAE: {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")

# ----------------------------
# 多子图绘制
# ----------------------------
num_samples = 6
cols = 3
rows = (num_samples + cols - 1) // cols

fig, axs = plt.subplots(rows, cols, figsize=(15, 8))
axs = axs.flatten()

for i in range(num_samples):
    axs[i].plot(range(pred_len), trues_all[i], label="Ground Truth", linewidth=2)
    axs[i].plot(range(pred_len), preds_all[i], label="Predicted", linewidth=2)
    axs[i].set_title(f"Sample {i}")
    axs[i].set_xlabel("Day")
    axs[i].set_ylabel("Power")
    axs[i].legend()
    axs[i].grid(True)

for i in range(num_samples, len(axs)):
    fig.delaxes(axs[i])

plt.suptitle("Power Forecasting - Prediction vs Ground Truth", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("multi_prediction_vs_gt.png")
plt.show()
