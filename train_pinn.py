import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import time
import gc
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# ==================== 配置参数 ====================
config = {
    'data_path': 'enterprise_dataset/printer_enterprise_data.csv',
    'seq_len': 200,
    'batch_size': 2048,  # 增加批量大小以提高训练速度
    'gradient_accumulation_steps': 1,  # 梯度累积平衡内存和速度
    'model_dim': 256,    # 保持模型维度
    'num_heads': 8,      # 保持注意力头数
    'num_layers': 6,     # 保持Transformer层数
    'dim_feedforward': 1024,  # 保持前馈网络维度
    'dropout': 0.1,
    'lr': 5e-4,         # 增加学习率加速收敛
    'epochs': 10,       # 大幅减少训练轮数（快速观察）
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 0,
    'max_samples': 500000,  # 减少样本数量（快速处理）
}

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# ==================== 数据处理器类 ====================
class MemoryDataProcessor:
    """数据处理器 - 直接加载到内存，不使用memmap"""
    def __init__(self, data_path, seq_len, max_samples):
        self.data_path = data_path
        self.seq_len = seq_len
        self.max_samples = max_samples

        self.input_cols = ['ctrl_T_target', 'ctrl_speed_set', 'ctrl_heater_base']
        self.target_cols = ['temperature_C', 'vibration_disp_m', 'vibration_vel_m_s',
                           'motor_current_A', 'pressure_bar', 'acoustic_signal']

        print(f"🔄 开始处理数据...")
        print(f"🚀 数据将直接加载到内存中")
        print(f"⚙️  最大样本限制: {self.max_samples}")
        self.process_data()

    def process_data(self):
        """预处理数据并直接加载到内存"""
        df = pd.read_csv(self.data_path)
        print(f"✅ 原始数据加载: {df.shape}")

        # Pass 1: 计算统计量和收集样本索引
        print("📊 [Pass 1/2] 计算全局统计量 + 收集样本索引...")
        start_time = time.time()

        X_sum = np.zeros(len(self.input_cols), dtype=np.float64)
        X_sq_sum = np.zeros(len(self.input_cols), dtype=np.float64)
        Y_sum = np.zeros(len(self.target_cols), dtype=np.float64)
        Y_sq_sum = np.zeros(len(self.target_cols), dtype=np.float64)
        count = 0
        sample_indices = []

        grouped = df.groupby('machine_id')

        for machine_id, group in grouped:
            group = group.sort_values('timestamp').reset_index(drop=True)
            X_raw = group[self.input_cols].values
            Y_raw = group[self.target_cols].values

            total_len = len(group)
            if total_len < self.seq_len + 1:
                continue

            n_windows = total_len - self.seq_len

            for i in range(n_windows):
                if count >= self.max_samples:
                    break
                    
                x_win = X_raw[i:i + self.seq_len]
                y_win = Y_raw[i + self.seq_len]

                X_sum += x_win.mean(axis=0)
                X_sq_sum += (x_win ** 2).mean(axis=0)
                Y_sum += y_win
                Y_sq_sum += y_win ** 2

                sample_indices.append((machine_id, i))
                count += 1
            
            if count >= self.max_samples:
                break

        # 计算均值和标准差
        self.mean_X = X_sum / count
        self.var_X = (X_sq_sum / count) - (self.mean_X ** 2)
        self.mean_Y = Y_sum / count
        self.var_Y = (Y_sq_sum / count) - (self.mean_Y ** 2)

        self.var_X = np.maximum(self.var_X, 0)
        self.var_Y = np.maximum(self.var_Y, 0)

        self.std_X = np.sqrt(self.var_X)
        self.std_Y = np.sqrt(self.var_Y)

        self.std_X[self.std_X < 1e-8] = 1.0
        self.std_Y[self.std_Y < 1e-8] = 1.0

        self.total_samples = count
        print(f"   实际处理样本总数: {self.total_samples}")
        print(f"   耗时: {time.time() - start_time:.2f}s")

        # 划分数据集
        self.split_idx = int(self.total_samples * 0.8)
        self.train_len = self.split_idx
        self.val_len = self.total_samples - self.split_idx
        print(f"   训练集: {self.train_len}, 验证集: {self.val_len}")

        # 划分样本索引
        train_sample_indices = sample_indices[:self.split_idx]
        val_sample_indices = sample_indices[self.split_idx:]

        # Shuffle训练集索引
        print("🔀 对训练集样本索引做全局 shuffle...")
        train_sample_indices = list(train_sample_indices)
        rng.shuffle(train_sample_indices)

        # Pass 2: 直接加载到内存
        print("📊 [Pass 2/2] 直接加载到内存...")
        start_time = time.time()

        # 初始化内存数组
        self.train_X = np.zeros((self.train_len, self.seq_len, len(self.input_cols)), dtype=np.float32)
        self.train_Y = np.zeros((self.train_len, len(self.target_cols)), dtype=np.float32)
        self.val_X = np.zeros((self.val_len, self.seq_len, len(self.input_cols)), dtype=np.float32)
        self.val_Y = np.zeros((self.val_len, len(self.target_cols)), dtype=np.float32)

        train_ptr = 0
        val_ptr = 0

        def write_samples(sample_indices_list, is_train):
            nonlocal train_ptr, val_ptr

            sorted_indices = sorted(sample_indices_list, key=lambda x: x[0])
            current_machine_id = None
            group_X = None
            group_Y = None

            for (mid, start) in sorted_indices:
                if mid != current_machine_id:
                    sub_df = df[df['machine_id'] == mid].sort_values('timestamp').reset_index(drop=True)
                    group_X = sub_df[self.input_cols].values
                    group_Y = sub_df[self.target_cols].values
                    current_machine_id = mid

                x_win = group_X[start:start + self.seq_len]
                y_win = group_Y[start + self.seq_len]

                x_norm = (x_win - self.mean_X) / self.std_X
                y_norm = (y_win - self.mean_Y) / self.std_Y

                if is_train:
                    self.train_X[train_ptr] = x_norm.astype(np.float32)
                    self.train_Y[train_ptr] = y_norm.astype(np.float32)
                    train_ptr += 1
                else:
                    self.val_X[val_ptr] = x_norm.astype(np.float32)
                    self.val_Y[val_ptr] = y_norm.astype(np.float32)
                    val_ptr += 1

        write_samples(train_sample_indices, is_train=True)
        write_samples(val_sample_indices, is_train=False)

        print(f"✅ 数据加载完成！内存占用: {self.train_X.nbytes / (1024**3):.2f} GB")
        print(f"   耗时: {time.time() - start_time:.2f}s")

    def inverse_transform_y(self, y_norm):
        """反归一化"""
        return y_norm * self.std_Y + self.mean_Y


class MemoryDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


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


class TimeSeriesTransformer(nn.Module):
    """时间序列Transformer模型"""
    def __init__(self, input_dim, output_dim, seq_len=200):
        super(TimeSeriesTransformer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        
        # 输入投影层
        self.input_proj = nn.Linear(input_dim, config['model_dim'])
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(config['model_dim'], seq_len)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['model_dim'],
            nhead=config['num_heads'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])
        
        # 输出层
        self.fc = nn.Linear(config['model_dim'], output_dim)

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


# ==================== 时间格式化函数 ====================
def format_time(seconds):
    """格式化时间为 HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# ==================== 训练与可视化 ====================
def visualize_predictions(model, loader, processor):
    """可视化预测结果"""
    import matplotlib.pyplot as plt

    model.eval()
    with torch.no_grad():
        for batch_X, batch_Y in loader:
            batch_X, batch_Y = batch_X.to(config['device']), batch_Y.to(config['device'])
            preds = model(batch_X)
            break

    preds_np = preds.cpu().numpy()
    targets_np = batch_Y.cpu().numpy()

    preds_real = processor.inverse_transform_y(preds_np)
    targets_real = processor.inverse_transform_y(targets_np)

    plt.figure(figsize=(12, 8))
    for i in range(6):
        plt.subplot(3, 2, i + 1)
        plt.plot(targets_real[:100, i], label='Ground Truth', alpha=0.7)
        plt.plot(preds_real[:100, i], label='Prediction', linestyle='--')
        plt.title(f'Feature: {processor.target_cols[i]}')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()

    os.makedirs('image', exist_ok=True)
    image_path = os.path.join('image', 'prediction_visualization.png')
    plt.savefig(image_path)
    print(f"📊 可视化图片已保存至: {image_path}")
    plt.show()


def train_model():
    print("=" * 70)
    print("🚀 TimeSeriesTransformer 快速训练 (10轮)")
    print("=" * 70)

    # 数据加载
    processor = MemoryDataProcessor(
        config['data_path'],
        config['seq_len'],
        config['max_samples']
    )

    train_dataset = MemoryDataset(processor.train_X, processor.train_Y)
    val_dataset = MemoryDataset(processor.val_X, processor.val_Y)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    input_dim = len(processor.input_cols)
    output_dim = len(processor.target_cols)

    # 使用完整的Transformer模型
    model = TimeSeriesTransformer(input_dim, output_dim, config['seq_len'])

    if torch.cuda.device_count() > 1:
        print(f"🎮 使用 {torch.cuda.device_count()} 个 GPU!")
        model = nn.DataParallel(model)

    model = model.to(config['device'])

    base_lr = config['lr']
    scaled_lr = base_lr

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        betas=(0.9, 0.999),
        weight_decay=1e-5
    )

    warmup_epochs = 3  # 减少warmup轮数
    total_epochs = config['epochs']

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    criterion = nn.MSELoss()
    scaler = GradScaler('cuda') 

    # TensorBoard设置
    log_dir = os.path.join("runs", "transformer_experiment_fast")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"📊 TensorBoard 日志目录: {log_dir}")

    best_val_loss = float('inf')
    global_step = 0
    print_every = 50  # 减少打印频率
    training_start_time = time.time()
    epoch_times = []
    
    print("\n🚀 开始训练...\n")
    
    for epoch in range(total_epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()  # 重置梯度

        # ==================== 训练循环 ====================
        for batch_idx, (batch_X, batch_Y) in enumerate(train_loader):
            batch_X, batch_Y = batch_X.to(config['device']), batch_Y.to(config['device'])

            with autocast('cuda'):
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y) / config['gradient_accumulation_steps']
            
            scaler.scale(loss).backward()
            
            # 梯度累积
            if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * config['gradient_accumulation_steps']

            if (batch_idx + 1) % print_every == 0:
                avg_so_far = epoch_loss / (batch_idx + 1)
                
                # 计算剩余时间
                elapsed_batch_time = time.time() - epoch_start
                batches_per_epoch = len(train_loader)
                batches_done = batch_idx + 1
                batches_remaining_in_epoch = batches_per_epoch - batches_done
                epochs_remaining = total_epochs - epoch - 1
                
                avg_batch_time = elapsed_batch_time / batches_done
                epoch_time_remaining = avg_batch_time * batches_remaining_in_epoch
                
                if epoch_times:
                    avg_epoch_time = sum(epoch_times) / len(epoch_times)
                    total_time_remaining = epoch_time_remaining + (avg_epoch_time * epochs_remaining)
                else:
                    total_time_remaining = epoch_time_remaining + (avg_batch_time * batches_per_epoch * epochs_remaining)
                
                # 打印日志
                print(f"  🔵 Epoch {epoch+1} | Batch {batch_idx+1:6d}/{len(train_loader):6d} | "
                      f"Loss: {avg_so_far:.6f} | LR: {optimizer.param_groups[0]['lr']:.6e} | "
                      f"Grad Norm: {total_norm:.4f} | "
                      f"ETA: {format_time(total_time_remaining)}")

                # TensorBoard 记录
                writer.add_scalar("Loss/train_batch", avg_so_far, global_step)
                writer.add_scalar("LR", optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar("Gradients/total_norm", total_norm, global_step)
                writer.add_scalar("Time/eta_seconds", total_time_remaining, global_step)
                
                global_step += 1

        avg_train_loss = epoch_loss / len(train_loader)
        writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)

        # ==================== 验证循环 ====================
        model.eval()
        val_loss = 0
        val_start_time = time.time()
        
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                batch_X, batch_Y = batch_X.to(config['device']), batch_Y.to(config['device'])
                with autocast('cuda'):
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_Y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_time = time.time() - val_start_time
        writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch)
        writer.add_scalar("Time/val_seconds", val_time, epoch)

        scheduler.step()

        # Epoch 统计
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        elapsed_total_time = time.time() - training_start_time
        epochs_completed = epoch + 1
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = total_epochs - epochs_completed
        estimated_remaining_time = avg_epoch_time * remaining_epochs
        
        progress_percent = (epochs_completed / total_epochs) * 100
        
        print(f"🟢 Epoch {epoch+1:3d}/{total_epochs} ({progress_percent:5.1f}%) | "
              f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
              f"Time: {epoch_time:.2f}s | 总用时: {format_time(elapsed_total_time)} | "
              f"ETA: {format_time(estimated_remaining_time)}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_transformer_model_fast.pth')
            print(f"  💾 模型已保存 (最佳验证损失: {best_val_loss:.6f})")

    total_training_time = time.time() - training_start_time
    print(f"\n{'='*70}")
    print(f"🎉 快速训练完成！")
    print(f"{'='*70}")
    print(f"⏱️  总用时: {format_time(total_training_time)}")
    print(f"📊 最佳验证损失: {best_val_loss:.6f}")
    print(f"{'='*70}\n")
    
    writer.close()

    print("📊 生成预测可视化图表...")
    visualize_predictions(model, val_loader, processor)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    train_model()
