# evaluate_pinn.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from train_pinn_seq2seq import (
    Config, PrinterPINN_Seq2Seq, MemoryDataProcessor, 
    PositionalEncoding, seq2seq_collate_fn
)
from torch.utils.data import DataLoader

def visualize_predictions(preds, targets, feature_names, save_path):
    """可视化预测结果"""
    fig, axes = plt.subplots(len(feature_names), 1, figsize=(12, 3*len(feature_names)))
    
    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        # 绘制第一条样本的预测
        pred_line = ax.plot(preds[0, :, i], label='Prediction', linestyle='--', linewidth=2)
        true_line = ax.plot(targets[0, :, i], label='Ground Truth', alpha=0.7, linewidth=2)
        
        # 添加置信区间 (简单的标准差)
        std = np.std(preds[:, :, i], axis=0)
        ax.fill_between(range(len(preds[0, :, i])), 
                       preds[0, :, i] - std, 
                       preds[0, :, i] + std, 
                       alpha=0.2, color=pred_line[0].get_color())
        
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.xlabel('Time Step (Future)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 可视化已保存: {save_path}")
    plt.close()

def calculate_metrics(preds, targets, feature_names):
    """计算评估指标"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics = {}
    batch_size, seq_len, n_features = preds.shape
    
    for i, name in enumerate(feature_names):
        pred_flat = preds[:, :, i].flatten()
        target_flat = targets[:, :, i].flatten()
        
        mse = mean_squared_error(target_flat, pred_flat)
        mae = mean_absolute_error(target_flat, pred_flat)
        r2 = r2_score(target_flat, pred_flat)
        
        metrics[name] = {
            'MSE': mse,
            'MAE': mae,
            'R2': r2,
            'RMSE': np.sqrt(mse)
        }
    
    return metrics

def fault_detection_analysis(preds, targets, threshold=2.0):
    """简单的故障检测分析"""
    errors = np.abs(preds - targets)
    max_errors = np.max(errors, axis=1)  # [batch_size, n_features]
    
    # 标记异常 (误差超过阈值倍的标准差)
    mean_errors = np.mean(errors, axis=1)
    std_errors = np.std(errors, axis=1)
    
    anomalies = (errors > mean_errors[:, np.newaxis, :] + threshold * std_errors[:, np.newaxis, :])
    
    return anomalies, max_errors

def evaluate_model(config_path, model_path, num_samples=100):
    """评估模型"""
    print("=" * 70)
    print("🔍 评估 Seq2Seq 模型")
    print("=" * 70)
    
    # 加载配置
    print(f"📂 加载配置: {config_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    config = Config()
    config.__dict__.update(checkpoint['config'])
    
    # 创建输出目录
    os.makedirs('evaluation_results', exist_ok=True)
    
    # 加载数据
    print("📊 加载数据...")
    processor = MemoryDataProcessor(
        config.data_path,
        config.seq_len,
        config.pred_len,
        config.max_samples,
        config
    )
    
    # 创建数据加载器
    val_dataset = type('Dataset', (), {
        '__len__': lambda self: len(processor.val_X),
        '__getitem__': lambda self, idx: (
            processor.val_X[idx], 
            processor.val_ctrl[idx], 
            processor.val_Y[idx]
        )
    })()
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        collate_fn=seq2seq_collate_fn
    )
    
    # 加载模型
    print("🤖 加载模型...")
    model = PrinterPINN_Seq2Seq(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.device)
    model.eval()
    
    # 预测
    print("🔮 进行预测...")
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x_hist, batch_x_ctrl, batch_y in val_loader:
            batch_x_hist = batch_x_hist.to(config.device)
            batch_x_ctrl = batch_x_ctrl.to(config.device)
            
            outputs = model(batch_x_hist, batch_x_ctrl)
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.numpy())
    
    # 合并结果
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # 反归一化
    preds_real = processor.inverse_transform_y(preds)
    targets_real = processor.inverse_transform_y(targets)
    
    # 计算指标
    print("📊 计算评估指标...")
    metrics = calculate_metrics(preds_real, targets_real, config.state_cols)
    
    # 打印结果
    print("\n" + "=" * 70)
    print("📊 评估结果")
    print("=" * 70)
    
    for feature, metric_dict in metrics.items():
        print(f"\n{feature}:")
        print(f"  MSE:  {metric_dict['MSE']:.6f}")
        print(f"  RMSE: {metric_dict['RMSE']:.6f}")
        print(f"  MAE:  {metric_dict['MAE']:.6f}")
        print(f"  R²:   {metric_dict['R2']:.6f}")
    
    # 故障检测分析
    print("\n" + "=" * 70)
    print("🔧 故障检测分析")
    print("=" * 70)
    
    anomalies, max_errors = fault_detection_analysis(preds_real, targets_real)
    
    print(f"异常样本数 (阈值=2σ): {np.any(anomalies, axis=(1,2)).sum()} / {len(anomalies)}")
    
    for i, feature in enumerate(config.state_cols):
        feature_anomalies = np.any(anomalies[:, :, i], axis=1)
        print(f"{feature}: {feature_anomalies.sum()} 异常时间步")
    
    # 可视化
    print("\n📊 生成可视化...")
    visualize_path = 'evaluation_results/prediction_comparison.png'
    visualize_predictions(preds_real, targets_real, config.state_cols, visualize_path)
    
    # 保存结果
    results_path = 'evaluation_results/metrics.txt'
    with open(results_path, 'w') as f:
        f.write("Model Evaluation Results\n")
        f.write("=" * 70 + "\n\n")
        
        for feature, metric_dict in metrics.items():
            f.write(f"{feature}:\n")
            for metric_name, value in metric_dict.items():
                f.write(f"  {metric_name}: {value:.6f}\n")
            f.write("\n")
    
    print(f"\n💾 结果已保存: {results_path}")
    print("=" * 70)

if __name__ == "__main__":
    # 指定模型路径
    model_path = "checkpoints_seq2seq/best_seq2seq_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先运行训练脚本 train_pinn_seq2seq.py")
        sys.exit(1)
    
    # 开始评估
    evaluate_model(None, model_path, num_samples=100)
