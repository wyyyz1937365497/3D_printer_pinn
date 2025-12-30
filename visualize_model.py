#!/usr/bin/env python3
"""
Transformer PINN 模型评估与可视化脚本（内存直接读取版）
支持直接从内存读取数据，不依赖硬盘缓存文件
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import gc
import argparse
from torch.utils.data import DataLoader
from torch.amp import autocast

# ==================== 配置参数 ====================
class Config:
    def __init__(self):
        self.seq_len = 200
        self.input_cols = ['ctrl_T_target', 'ctrl_speed_set', 'ctrl_heater_base']
        self.target_cols = ['temperature_C', 'vibration_disp_m', 'vibration_vel_m_s',
                          'motor_current_A', 'pressure_bar', 'acoustic_signal']

# ==================== 模型定义 ====================
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class PrinterPINN(nn.Module):
    """3D打印机物理信息神经网络"""
    def __init__(self, input_dim, output_dim, seq_len=200):
        super(PrinterPINN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        
        # 输入投影层
        self.input_proj = nn.Linear(input_dim, 256)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(256, seq_len)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # 输出层
        self.fc = nn.Linear(256, output_dim)

    def forward(self, x):
        # 输入投影
        x = self.input_proj(x)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Transformer处理
        x = self.transformer(x)
        
        # 取最后一个时间步的输出
        x = x[:, -1, :]
        
        # 输出层
        prediction = self.fc(x)
        return prediction

# ==================== 数据加载器 ====================
class MemoryDataset(torch.utils.data.Dataset):
    """直接从内存加载数据的Dataset"""
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ==================== 主函数 ====================
def load_model_and_visualize(
    model_path='best_pinn_model.pth',
    cache_dir='./data_cache/',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    batch_size=1024,
    num_samples_to_plot=200,
    save_path='image/pinn_prediction_visualization.png',
    save_metrics='image/pinn_metrics_report.txt'
):
    """
    加载 Transformer PINN 模型并生成可视化图表
    
    参数:
        model_path: 模型权重文件路径
        cache_dir: 数据缓存目录（用于归一化参数）
        device: 运行设备 ('cuda' 或 'cpu')
        batch_size: 批次大小
        num_samples_to_plot: 可视化时显示的样本数量
        save_path: 图片保存路径
        save_metrics: 指标报告保存路径
    """
    print("=" * 70)
    print("🎨 Transformer PINN 模型可视化脚本（内存直接读取版）")
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
    
    # 4. 直接从内存加载数据（这里需要根据你的实际数据加载方式修改）
    # 假设数据已经加载到内存中，可以通过某种方式获取
    # 你需要根据实际的数据加载方式修改这部分代码
    print("\n📂 直接从内存加载数据...")
    
    # 示例：假设数据已经加载到全局变量中
    # 实际使用时，你需要根据你的数据加载方式获取 X 和 Y
    try:
        # 尝试从全局变量或内存中获取数据
        # 这里需要根据你的实际数据加载方式修改
        import sys
        if 'val_X' in sys.modules['__main__'].__dict__:
            val_X = sys.modules['__main__'].__dict__['val_X']
            val_Y = sys.modules['__main__'].__dict__['val_Y']
            print("   从全局变量加载验证数据")
        else:
            raise ImportError("无法从全局变量获取数据，请确保数据已正确加载")
    except Exception as e:
        raise RuntimeError(f"数据加载失败: {e}")
    
    print(f"   验证数据形状: X={val_X.shape}, Y={val_Y.shape}")
    
    # 5. 创建数据加载器
    val_dataset = MemoryDataset(val_X, val_Y)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # 6. 创建模型
    print("\n🏗️  创建 Transformer PINN 模型...")
    input_dim = len(mean_X)
    output_dim = len(mean_Y)
    
    model = PrinterPINN(input_dim, output_dim)
    
    # 7. 加载模型权重
    print(f"📦 加载模型权重: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 检查是否是完整的检查点文件
    if 'model_state_dict' in checkpoint:
        # 完整的检查点文件，提取模型状态字典
        state_dict = checkpoint['model_state_dict']
        print("   检测到完整的检查点文件，提取模型状态字典")
    else:
        # 纯模型权重文件
        state_dict = checkpoint
    
    # 处理 DataParallel 模型
    if 'module.' in list(state_dict.keys())[0]:
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    print("✅ 模型加载完成")
    
    # 8. 进行预测
    print("\n🔮 开始预测...")
    all_preds = []
    all_targets = []
    total_loss = 0.0
    criterion = nn.MSELoss()
    
    use_amp = device == 'cuda'
    
    with torch.no_grad():
        for batch_idx, (batch_X, batch_Y) in enumerate(val_loader):
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            
            if use_amp:
                with autocast('cuda'):
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_Y)
            else:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)
            
            total_loss += loss.item() * batch_X.size(0)
            
            # 保存结果并转移到 CPU
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_Y.cpu().numpy())
            
            # 清理 GPU 内存
            del outputs, loss, batch_X, batch_Y
            if device == 'cuda' and batch_idx % 5 == 0:
                torch.cuda.empty_cache()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"   Batch {batch_idx+1}/{len(val_loader)}")
    
    # 9. 合并所有预测
    print("\n🔄 合并预测结果...")
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    
    # 清理内存
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
    
    # 清理中间变量
    del preds, targets
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    # 12. 计算各特征指标
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
        
        # 计算 R²
        ss_res = np.sum((target_i - pred_i) ** 2)
        ss_tot = np.sum((target_i - np.mean(target_i)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # 计算 MAPE（避免除零）
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
        f.write("Transformer PINN 模型性能指标报告\n")
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

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Evaluate Transformer PINN model')
    parser.add_argument('--model_path', type=str, default='best_pinn_model.pth',
                       help='Path to the model weights file (default: best_pinn_model.pth)')
    parser.add_argument('--cache_dir', type=str, default='./data_cache/',
                       help='Directory containing cached data (default: ./data_cache/)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='Batch size for evaluation (default: 1024)')
    parser.add_argument('--num_samples', type=int, default=200,
                       help='Number of samples to plot (default: 200)')
    parser.add_argument('--save_path', type=str, default='image/pinn_prediction_visualization.png',
                       help='Path to save visualization image (default: image/pinn_prediction_visualization.png)')
    parser.add_argument('--metrics_path', type=str, default='image/pinn_metrics_report.txt',
                       help='Path to save metrics report (default: image/pinn_metrics_report.txt)')
    
    args = parser.parse_args()
    
    # 调用评估函数
    metrics, val_loss = load_model_and_visualize(
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_samples_to_plot=args.num_samples,
        save_path=args.save_path,
        save_metrics=args.metrics_path
    )

if __name__ == "__main__":
    main()
