# visualize_model.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from collections import OrderedDict
from torch.utils.data import DataLoader
import sklearn.metrics as metrics

# 从训练脚本导入必要的类
# 确保同级目录下有 train_pinn_seq2seq.py
from train_pinn_seq2seq import (
    Config, PrinterPINN_Seq2Seq, MemoryDataProcessor, 
    PositionalEncoding, seq2seq_collate_fn
)

# ==================== 评估与可视化函数 ====================

def load_model_checkpoint(model_path, model, device='cpu'):
    """
    加载模型权重，自动处理 DataParallel 的 'module.' 前缀问题
    """
    print(f"📂 正在加载模型检查点: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # 创建一个新的 state_dict，去掉 'module.' 前缀
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k # 移除 'module.'
        new_state_dict[name] = v
    
    # 加载修正后的权重
    model.load_state_dict(new_state_dict)
    print("✅ 模型权重加载成功（已处理 DataParallel 前缀）")
    
    # 返回配置信息（如果有）
    config_dict = checkpoint.get('config', {})
    return config_dict

def visualize_predictions(preds, targets, feature_names, save_path, processor):
    """可视化预测结果，处理不同量纲"""
    fig, axes = plt.subplots(len(feature_names), 1, figsize=(14, 2.5*len(feature_names)))
    
    # 如果只有一个特征，axes不是数组，需转换
    if len(feature_names) == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        # 获取真实范围用于绘图
        true_vals = targets[0, :, i]
        pred_vals = preds[0, :, i]
        
        # 绘制
        ax.plot(true_vals, label='Ground Truth', alpha=0.8, linewidth=2, color='tab:blue')
        ax.plot(pred_vals, label='Prediction', linestyle='--', linewidth=2, color='tab:orange')
        
        # 计算误差并填充
        error = np.abs(pred_vals - true_vals)
        # ax.fill_between(range(len(pred_vals)), pred_vals - error, pred_vals + error, alpha=0.2, color='tab:orange')
        
        # 设置标题和标签
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 计算该指标的平均绝对误差并显示在图上
        mae = np.mean(np.abs(preds[:, :, i] - targets[:, :, i]))
        ax.text(0.02, 0.9, f'MAE: {mae:.4f}', transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.7))
    
    plt.xlabel('Time Step (Future)', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 可视化已保存: {save_path}")
    plt.close()

def calculate_metrics(preds, targets, feature_names):
    """计算评估指标"""
    metrics_results = {}
    batch_size, seq_len, n_features = preds.shape
    
    for i, name in enumerate(feature_names):
        pred_flat = preds[:, :, i].flatten()
        target_flat = targets[:, :, i].flatten()
        
        # 忽略 NaN 或 Inf
        mask = np.isfinite(pred_flat) & np.isfinite(target_flat)
        if mask.sum() == 0:
            print(f"⚠️  特征 {name} 包含无效数据，跳过评估。")
            continue

        mse = metrics.mean_squared_error(target_flat[mask], pred_flat[mask])
        mae = metrics.mean_absolute_error(target_flat[mask], pred_flat[mask])
        
        # R2 score 可能为负，说明模型极差
        try:
            r2 = metrics.r2_score(target_flat[mask], pred_flat[mask])
        except:
            r2 = -999.0
            
        metrics_results[name] = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': np.sqrt(mse),
            'R2': r2
        }
    
    return metrics_results

def evaluate_model(config_path, model_path, num_samples=100):
    """主评估函数"""
    print("=" * 70)
    print("🔍 评估 Seq2Seq 模型")
    print("=" * 70)
    
    # 1. 加载检查点以获取配置
    # 为了加载配置，我们需要先有一个临时的 Config 对象
    # 实际上 Config 是定义在 train 脚本里的，我们直接实例化即可
    temp_config = Config()
    
    try:
        loaded_config = load_model_checkpoint(model_path, PrinterPINN_Seq2Seq(temp_config))
        # 如果检查点里有保存的配置，更新当前配置
        if loaded_config:
            # 将字典更新到 Config 对象中
            for k, v in loaded_config.items():
                if hasattr(temp_config, k):
                    setattr(temp_config, k, v)
        print(f"✅ 已从检查点恢复配置参数")
    except Exception as e:
        print(f"⚠️  无法从检查点恢复配置，使用默认配置: {e}")
    
    # 2. 准备输出目录
    os.makedirs('evaluation_results', exist_ok=True)
    
    # 3. 加载数据 (使用恢复的配置)
    print("\n📊 加载并处理数据...")
    try:
        processor = MemoryDataProcessor(
            temp_config.data_path,
            temp_config.seq_len,
            temp_config.pred_len,
            temp_config.max_samples,
            temp_config
        )
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        print("提示: 确保数据路径正确且列名与训练时一致。")
        return

    # 创建验证集 Loader
    # 定义一个简单的 Dataset 包装器，避免重新定义类
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
        batch_size=32, # 评估时 batch size 可以小一点，防止 OOM
        shuffle=False,
        num_workers=0,
        collate_fn=seq2seq_collate_fn
    )
    
    # 4. 初始化并加载模型
    print("🤖 初始化并加载模型...")
    model = PrinterPINN_Seq2Seq(temp_config)
    model = model.to(temp_config.device)
    
    # 关键步骤：加载权重（去掉 module.）
    load_model_checkpoint(model_path, model, temp_config.device)
    
    model.eval()
    
    # 5. 执行预测
    print("🔮 开始预测验证集...")
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x_hist, batch_x_ctrl, batch_y in val_loader:
            batch_x_hist = batch_x_hist.to(temp_config.device)
            batch_x_ctrl = batch_x_ctrl.to(temp_config.device)
            
            outputs = model(batch_x_hist, batch_x_ctrl)
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.numpy())
    
    # 合并结果
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # 6. 反归一化
    print("📈 反归一化数据...")
    preds_real = processor.inverse_transform_y(preds)
    targets_real = processor.inverse_transform_y(targets)
    
    # 7. 计算指标
    print("📊 计算评估指标...")
    metrics_res = calculate_metrics(preds_real, targets_real, temp_config.state_cols)
    
    # 打印结果
    print("\n" + "=" * 70)
    print("📊 评估结果")
    print("=" * 70)
    
    for feature, metric_dict in metrics_res.items():
        print(f"\n【{feature}】")
        print(f"  MSE:  {metric_dict['MSE']:.6f}")
        print(f"  RMSE: {metric_dict['RMSE']:.6f}")
        print(f"  MAE:  {metric_dict['MAE']:.6f}")
        print(f"  R²:   {metric_dict['R2']:.6f}")
    
    # 8. 可视化
    print("\n📊 生成可视化图表...")
    visualize_path = 'evaluation_results/prediction_visualization.png'
    # 只取前 num_samples 个样本进行可视化，避免图表过于密集
    visualize_predictions(
        preds_real[:num_samples], 
        targets_real[:num_samples], 
        temp_config.state_cols, 
        visualize_path,
        processor
    )
    
    # 9. 保存结果到文本文件
    results_path = 'evaluation_results/metrics_report.txt'
    with open(results_path, 'w') as f:
        f.write("Model Evaluation Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Samples Evaluated: {len(preds_real)}\n\n")
        
        for feature, metric_dict in metrics_res.items():
            f.write(f"Feature: {feature}\n")
            for metric_name, value in metric_dict.items():
                f.write(f"  {metric_name}: {value:.6f}\n")
            f.write("\n")
            
    print(f"💾 详细报告已保存: {results_path}")
    print("=" * 70)
    print("✅ 评估完成！")

if __name__ == "__main__":
    # 默认模型路径
    # 如果你在训练中使用了 DataParallel，这里会自动处理
    model_path = "checkpoints_seq2seq/best_seq2seq_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ 错误: 找不到模型文件 {model_path}")
        print("请检查 train_pinn_seq2seq.py 中的 checkpoint_dir 设置")
        sys.exit(1)
    
    # 开始评估
    evaluate_model(None, model_path, num_samples=100)
