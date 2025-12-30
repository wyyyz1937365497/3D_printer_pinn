#!/usr/bin/env python3
"""
独立的模型测试与可视化脚本（修复版）
修复 CUDA OOM 问题：减小 batch_size 并优化内存使用
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import gc

# ==================== 模型定义 ====================
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.kernel_size = kernel_size
        self.dilation = dilation

    def forward(self, x):
        out = self.net(x)
        pad = (self.kernel_size - 1) * self.dilation
        out = out[:, :, :-pad]
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                   dilation=dilation_size, padding=padding, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.network(x)
        return out.transpose(1, 2)


class TCNLSTMModel(nn.Module):
    def __init__(self, input_dim, tcn_channels, hidden_dim, output_dim):
        super(TCNLSTMModel, self).__init__()
        self.tcn = TCN(input_dim, tcn_channels)
        tcn_output_dim = tcn_channels[-1]
        self.lstm = nn.LSTM(tcn_output_dim, hidden_dim, num_layers=2,
                           batch_first=True, dropout=0.1, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        tcn_out = self.tcn(x)
        lstm_out, (h_n, c_n) = self.lstm(tcn_out)
        last_step_out = lstm_out[:, -1, :]
        prediction = self.fc(last_step_out)
        return prediction


# ==================== 数据加载器 ====================
class SimpleMMapDataset(torch.utils.data.Dataset):
    def __init__(self, X_mmap, Y_mmap):
        self.X = X_mmap
        self.Y = Y_mmap

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx].copy(), self.Y[idx].copy()


# ==================== 主函数 ====================
def load_model_and_visualize(
    model_path='best_tcn_lstm_model.pth',
    cache_dir='./data_cache/',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    batch_size=1024,  # 🔧 减小 batch_size 从 8192 -> 1024
    num_samples_to_plot=200,
    save_path='image/prediction_visualization.png',
    save_metrics='image/metrics_report.txt'
):
    """
    加载模型并生成可视化图表（修复 OOM 问题）
    
    参数:
        model_path: 模型权重文件路径
        cache_dir: 数据缓存目录
        device: 运行设备 ('cuda' 或 'cpu')
        batch_size: 批次大小（减小以避免 OOM）
        num_samples_to_plot: 可视化时显示的样本数量
        save_path: 图片保存路径
        save_metrics: 指标报告保存路径
    """
    print("=" * 70)
    print("🎨 模型可视化脚本（修复版）")
    print("=" * 70)
    
    # 1. 检查文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 模型文件不存在: {model_path}")
    if not os.path.exists(cache_dir):
        raise FileNotFoundError(f"❌ 缓存目录不存在: {cache_dir}")
    
    print(f"✅ 模型文件: {model_path}")
    print(f"✅ 缓存目录: {cache_dir}")
    print(f"✅ 设备: {device}")
    print(f"✅ Batch Size: {batch_size}")
    
    # 2. 清理 GPU 缓存
    if device == 'cuda':
        torch.cuda.empty_cache()
        print("🧹 已清理 GPU 缓存")
    
    # 3. 加载归一化参数
    print("\n📊 加载归一化参数...")
    scaler_path = os.path.join(cache_dir, 'scaler_stats.npz')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"❌ 找不到归一化参数文件: {scaler_path}")
    
    scaler_data = np.load(scaler_path)
    mean_X = scaler_data['mean_X']
    std_X = scaler_data['std_X']
    mean_Y = scaler_data['mean_Y']
    std_Y = scaler_data['std_Y']
    
    print(f"   Input mean: {mean_X}")
    print(f"   Input std:  {std_X}")
    print(f"   Target mean: {mean_Y}")
    print(f"   Target std:  {std_Y}")
    
    # 4. 加载验证数据
    print("\n📂 加载验证数据...")
    val_X = np.load(os.path.join(cache_dir, 'val_X.npy'), mmap_mode='r')
    val_Y = np.load(os.path.join(cache_dir, 'val_Y.npy'), mmap_mode='r')
    
    print(f"   验证数据形状: X={val_X.shape}, Y={val_Y.shape}")
    
    # 5. 创建数据加载器
    val_dataset = SimpleMMapDataset(val_X, val_Y)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False  # 🔧 关闭 pin_memory 以减少内存占用
    )
    
    # 6. 创建模型
    print("\n🏗️  创建模型...")
    input_dim = len(mean_X)
    output_dim = len(mean_Y)
    tcn_channels = [64, 64, 128]
    hidden_dim = 128
    
    model = TCNLSTMModel(input_dim, tcn_channels, hidden_dim, output_dim)
    
    # 7. 加载模型权重
    print(f"📦 加载模型权重: {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    
    # 如果模型是用 DataParallel 保存的，去掉 'module.' 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    
    model = model.to(device)
    model.eval()
    print("✅ 模型加载完成")
    
    # 8. 进行预测
    print("\n🔮 开始预测...")
    all_preds = []
    all_targets = []
    total_loss = 0.0
    criterion = nn.MSELoss()
    
    # 🔧 使用混合精度推理
    use_amp = device == 'cuda'
    
    with torch.no_grad():
        for batch_idx, (batch_X, batch_Y) in enumerate(val_loader):
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            
            # 🔧 使用 autocast 减少内存占用
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_Y)
            else:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)
            
            total_loss += loss.item() * batch_X.size(0)
            
            # 保存结果（立即转移到 CPU）
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_Y.cpu().numpy())
            
            # 🔧 立即清理 GPU 内存
            del outputs, loss, batch_X, batch_Y
            if device == 'cuda' and batch_idx % 5 == 0:
                torch.cuda.empty_cache()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"   Batch {batch_idx+1}/{len(val_loader)}")
    
    # 9. 合并所有预测
    print("\n🔄 合并预测结果...")
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    
    # 🔧 清理列表以释放内存
    del all_preds, all_targets
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    # 10. 计算平均损失
    avg_loss = total_loss / len(val_dataset)
    print(f"📊 验证集平均损失: {avg_loss:.6f}")
    
    # 11. 反归一化
    print("🔄 反归一化...")
    preds_real = preds * std_Y + mean_Y
    targets_real = targets * std_Y + mean_Y
    
    # 🔧 清理中间变量
    del preds, targets
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    # 12. 计算每个特征的指标
    print("\n📈 计算各特征指标...")
    feature_names = ['temperature_C', 'vibration_disp_m', 'vibration_vel_m_s',
                     'motor_current_A', 'pressure_bar', 'acoustic_signal']
    
    metrics_list = []
    for i, name in enumerate(feature_names):
        pred_i = preds_real[:, i]
        target_i = targets_real[:, i]
        
        mse = np.mean((pred_i - target_i) ** 2)
        mae = np.mean(np.abs(pred_i - target_i))
        rmse = np.sqrt(mse)
        
        # 计算R²
        ss_res = np.sum((target_i - pred_i) ** 2)
        ss_tot = np.sum((target_i - np.mean(target_i)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # 计算MAPE（避免除零）
        mask = np.abs(target_i) > 1e-6
        mape = np.mean(np.abs((target_i[mask] - pred_i[mask]) / target_i[mask])) * 100 if np.any(mask) else 0
        
        metrics_list.append({
            'feature': name,
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'MAPE (%)': mape
        })
    
    # 13. 生成可视化
    print("\n🎨 生成可视化图表...")
    
    # 创建图片保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 大图：所有6个特征
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    n_plot = min(num_samples_to_plot, len(preds_real))
    
    for i, name in enumerate(feature_names):
        ax = axes[i]
        ax.plot(targets_real[:n_plot, i], label='Ground Truth', alpha=0.7, linewidth=2)
        ax.plot(preds_real[:n_plot, i], label='Prediction', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.set_title(f'{name}\nRMSE: {metrics_list[i]["RMSE"]:.4f} | R²: {metrics_list[i]["R²"]:.4f}', 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 可视化图表已保存: {save_path}")
    plt.show()
    
    # 14. 生成散点图（预测 vs 真实值）
    scatter_path = save_path.replace('.png', '_scatter.png')
    fig2, axes2 = plt.subplots(3, 2, figsize=(16, 12))
    axes2 = axes2.flatten()
    
    for i, name in enumerate(feature_names):
        ax = axes2[i]
        ax.scatter(targets_real[:, i], preds_real[:, i], alpha=0.1, s=1)
        ax.plot([targets_real[:, i].min(), targets_real[:, i].max()],
                [targets_real[:, i].min(), targets_real[:, i].max()], 
                'r--', linewidth=2, label='Perfect Prediction')
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Prediction')
        ax.set_title(f'{name} - Scatter Plot', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    print(f"✅ 散点图已保存: {scatter_path}")
    plt.show()
    
    # 15. 保存指标报告
    print("\n📝 保存指标报告...")
    os.makedirs(os.path.dirname(save_metrics), exist_ok=True)
    
    with open(save_metrics, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("模型性能指标报告\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"模型文件: {model_path}\n")
        f.write(f"验证集大小: {len(val_dataset)}\n")
        f.write(f"验证集平均损失: {avg_loss:.6f}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("各特征详细指标\n")
        f.write("-" * 70 + "\n\n")
        
        for metrics in metrics_list:
            f.write(f"特征: {metrics['feature']}\n")
            f.write(f"  MSE (均方误差):     {metrics['MSE']:.6f}\n")
            f.write(f"  MAE (平均绝对误差): {metrics['MAE']:.6f}\n")
            f.write(f"  RMSE (均方根误差):  {metrics['RMSE']:.6f}\n")
            f.write(f"  R² (决定系数):      {metrics['R²']:.6f}\n")
            f.write(f"  MAPE (平均绝对百分比误差): {metrics['MAPE (%)']:.2f}%\n\n")
    
    print(f"✅ 指标报告已保存: {save_metrics}")
    
    # 16. 打印摘要
    print("\n" + "=" * 70)
    print("📊 性能摘要")
    print("=" * 70)
    for metrics in metrics_list:
        print(f"{metrics['feature']}:")
        print(f"  RMSE: {metrics['RMSE']:.4f}, R²: {metrics['R²']:.4f}, MAPE: {metrics['MAPE (%)']:.2f}%")
    
    print("\n" + "=" * 70)
    print("✅ 可视化完成！")
    print("=" * 70)
    
    return metrics_list, avg_loss


if __name__ == "__main__":
    # 🔧 可以根据 GPU 内存情况调整 batch_size
    # 如果还是 OOM，可以继续减小到 512 或 256
    metrics, val_loss = load_model_and_visualize(
        model_path='best_tcn_lstm_model.pth',
        cache_dir='./data_cache/',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        batch_size=1024,  # 从 8192 减小到 1024（如果还是 OOM，改为 512 或 256）
        num_samples_to_plot=200,
        save_path='image/prediction_visualization.png',
        save_metrics='image/metrics_report.txt'
    )

