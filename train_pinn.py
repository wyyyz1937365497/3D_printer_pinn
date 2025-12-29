import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import time
import gc
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# ==================== 配置参数 ====================
config = {
    'data_path': 'enterprise_dataset/printer_enterprise_data.csv',
    'cache_dir': './data_cache/',
    'seq_len': 200,
    'batch_size': 4096,  # 可以在这里调整batch_size
    'hidden_dim': 128,
    'tcn_channels': [64, 64, 128],
    'lr': 2e-3,
    'epochs': 50,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 0,
    'test_mode': False,
    'test_samples': 1000,
}

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# ==================== 数据处理器类 ====================
# (这里保持原有的 ShuffledOnceDataProcessor 类不变)
# 为了简洁，我会在下面提供完整的修改

class ShuffledOnceDataProcessor:
    """数据处理器 - 与之前相同"""
    def __init__(self, data_path, seq_len, cache_dir, test_mode=False, test_samples=1000):
        self.data_path = data_path
        self.seq_len = seq_len
        self.cache_dir = cache_dir
        self.test_mode = test_mode
        self.test_samples = test_samples

        self.input_cols = ['ctrl_T_target', 'ctrl_speed_set', 'ctrl_heater_base']
        self.target_cols = ['temperature_C', 'vibration_disp_m', 'vibration_vel_m_s',
                           'motor_current_A', 'pressure_bar', 'acoustic_signal']

        if os.path.exists(cache_dir) and os.listdir(cache_dir):
            print(f"📦 发现缓存目录: {cache_dir}")
            self.load_metadata()
        else:
            print(f"🔄 缓存不存在，开始处理数据...")
            print(f"🚀 缓存将写入: {cache_dir}")
            os.makedirs(cache_dir, exist_ok=True)
            self.process_and_save()

    def process_and_save(self):
        """预处理数据并保存"""
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
                x_win = X_raw[i:i + self.seq_len]
                y_win = Y_raw[i + self.seq_len]

                X_sum += x_win.mean(axis=0)
                X_sq_sum += (x_win ** 2).mean(axis=0)
                Y_sum += y_win
                Y_sq_sum += y_win ** 2

                sample_indices.append((machine_id, i))
                count += 1

                if self.test_mode and count >= self.test_samples:
                    break

            if self.test_mode and count >= self.test_samples:
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
        print(f"   样本总数: {self.total_samples}")
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

        # 保存归一化参数
        print("💾 保存归一化参数")
        scaler_path = os.path.join(self.cache_dir, 'scaler_stats.npz')
        np.savez(scaler_path,
                 mean_X=self.mean_X, std_X=self.std_X,
                 mean_Y=self.mean_Y, std_Y=self.std_Y)
        print(f"   已保存至: {scaler_path}")

        # 保存样本索引
        indices_path = os.path.join(self.cache_dir, 'sample_indices.npz')
        np.savez(indices_path,
                 train_sample_indices=np.array(train_sample_indices, dtype=object),
                 val_sample_indices=np.array(val_sample_indices, dtype=object))
        print(f"   样本索引已保存至: {indices_path}")

        # Pass 2: 写入memmap
        print("💾 [Pass 2/2] 按 train/val 索引顺序写入 memmap 缓存文件...")

        mmap_files = {
            'train_X': os.path.join(self.cache_dir, 'train_X.npy'),
            'train_Y': os.path.join(self.cache_dir, 'train_Y.npy'),
            'val_X': os.path.join(self.cache_dir, 'val_X.npy'),
            'val_Y': os.path.join(self.cache_dir, 'val_Y.npy'),
        }

        self.train_X = np.lib.format.open_memmap(
            mmap_files['train_X'], dtype='float32', mode='w+',
            shape=(self.train_len, self.seq_len, len(self.input_cols))
        )
        self.train_Y = np.lib.format.open_memmap(
            mmap_files['train_Y'], dtype='float32', mode='w+',
            shape=(self.train_len, len(self.target_cols))
        )
        self.val_X = np.lib.format.open_memmap(
            mmap_files['val_X'], dtype='float32', mode='w+',
            shape=(self.val_len, self.seq_len, len(self.input_cols))
        )
        self.val_Y = np.lib.format.open_memmap(
            mmap_files['val_Y'], dtype='float32', mode='w+',
            shape=(self.val_len, len(self.target_cols))
        )

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

        print("✅ 缓存写入完成！")

        del self.train_X, self.train_Y, self.val_X, self.val_Y
        gc.collect()

        self.load_metadata()

    def load_metadata(self):
        """加载缓存的元数据"""
        scaler_path = os.path.join(self.cache_dir, 'scaler_stats.npz')
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"找不到 Scaler 文件: {scaler_path}")

        data = np.load(scaler_path)
        self.mean_X = data['mean_X']
        self.std_X = data['std_X']
        self.mean_Y = data['mean_Y']
        self.std_Y = data['std_Y']
        print("✅ 归一化参数加载成功")

        self.train_X = np.load(os.path.join(self.cache_dir, 'train_X.npy'), mmap_mode='r')
        self.train_Y = np.load(os.path.join(self.cache_dir, 'train_Y.npy'), mmap_mode='r')
        self.val_X = np.load(os.path.join(self.cache_dir, 'val_X.npy'), mmap_mode='r')
        self.val_Y = np.load(os.path.join(self.cache_dir, 'val_Y.npy'), mmap_mode='r')

        self.train_len = self.train_X.shape[0]
        self.val_len = self.val_X.shape[0]
        self.total_samples = self.train_len + self.val_len
        print(f"✅ 数据映射加载成功: Train {self.train_len}, Val {self.val_len}")

    def inverse_transform_y(self, y_norm):
        """反归一化"""
        return y_norm * self.std_Y + self.mean_Y


class MMapDataset(Dataset):
    def __init__(self, X_mmap, Y_mmap):
        self.X = X_mmap
        self.Y = Y_mmap

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx].copy(), self.Y[idx].copy()


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


# ==================== 时间格式化函数 ====================
def format_time(seconds):
    """格式化时间为 HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# ==================== 训练与可视化（带剩余时间预测） ====================
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
    print("🚀 TCN-LSTM 训练 (AMP + warmup+cosine + TensorBoard + 剩余时间预测)")
    print("=" * 70)

    # 数据加载
    processor = ShuffledOnceDataProcessor(
        config['data_path'],
        config['seq_len'],
        config['cache_dir'],
        test_mode=config['test_mode'],
        test_samples=config['test_samples']
    )

    train_dataset = MMapDataset(processor.train_X, processor.train_Y)
    val_dataset = MMapDataset(processor.val_X, processor.val_Y)

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

    model = TCNLSTMModel(input_dim, config['tcn_channels'], config['hidden_dim'], output_dim)

    if torch.cuda.device_count() > 1:
        print(f"🎮 使用 {torch.cuda.device_count()} 个 GPU!")
        model = nn.DataParallel(model)

    model = model.to(config['device'])

    # 🎯 关键修改：根据batch_size调整学习率
    base_lr = config['lr']
    scaled_lr = base_lr * (config['batch_size'] / 2048)  # 以2048为基准进行缩放
    scaled_lr = min(scaled_lr, base_lr * 4)  # 限制最大放大倍数，避免学习率过大
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=1e-5
    )

    # 学习率调度：warmup + cosine 退火
    warmup_epochs = 5
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
    scaler = GradScaler()

    # TensorBoard设置
    log_dir = os.path.join("runs", "tcn_lstm_experiment")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"📊 TensorBoard 日志目录: {log_dir}")
    print(f"   运行: tensorboard --logdir {log_dir}")

    # 📊 训练配置信息输出
    print("🏋️ 训练配置:")
    print(f"📝 学习率: {base_lr:.6f} (基准) -> {scaled_lr:.6f} (调整后)")
    print(f"📦 Batch Size: {config['batch_size']}")
    print(f"🧠 隐藏层维度: {config['hidden_dim']}")
    print(f"📊 总训练样本: {len(train_dataset)}")
    print(f"📊 总验证样本: {len(val_dataset)}")
    print(f"⏱️  总Epoch数: {total_epochs}")
    
    best_val_loss = float('inf')
    global_step = 0
    print_every = 100
    
    # ⏰ 时间追踪变量
    training_start_time = time.time()
    epoch_times = []
    
    print("\n🚀 开始训练...\n")
    
    for epoch in range(total_epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0

        # 📊 训练循环
        for batch_idx, (batch_X, batch_Y) in enumerate(train_loader):
            batch_X, batch_Y = batch_X.to(config['device']), batch_Y.to(config['device'])

            optimizer.zero_grad()

            with autocast():
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            # 📊 定期打印进度和剩余时间
            if (batch_idx + 1) % print_every == 0:
                avg_so_far = epoch_loss / (batch_idx + 1)
                
                # 实时计算剩余时间
                elapsed_batch_time = time.time() - epoch_start
                batches_per_epoch = len(train_loader)
                batches_done = batch_idx + 1
                batches_remaining_in_epoch = batches_per_epoch - batches_done
                epochs_remaining = total_epochs - epoch - 1
                
                # 估算当前epoch剩余时间
                avg_batch_time = elapsed_batch_time / batches_done
                epoch_time_remaining = avg_batch_time * batches_remaining_in_epoch
                
                # 估算总体剩余时间
                if epoch_times:
                    avg_epoch_time = sum(epoch_times) / len(epoch_times)
                    total_time_remaining = epoch_time_remaining + (avg_epoch_time * epochs_remaining)
                else:
                    total_time_remaining = epoch_time_remaining + (avg_batch_time * batches_per_epoch * epochs_remaining)
                
                print(f"  🔵 Epoch {epoch+1} | Batch {batch_idx+1:6d}/{len(train_loader):6d} | "
                      f"Loss: {avg_so_far:.6f} | LR: {optimizer.param_groups[0]['lr']:.6e} | "
                      f"ETA: {format_time(total_time_remaining)}")

                writer.add_scalar("Loss/train_batch", avg_so_far, global_step)
                writer.add_scalar("LR", optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar("Time/eta_seconds", total_time_remaining, global_step)
                global_step += 1

        avg_train_loss = epoch_loss / len(train_loader)
        writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)

        # 📊 验证循环
        model.eval()
        val_loss = 0
        val_start_time = time.time()
        
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                batch_X, batch_Y = batch_X.to(config['device']), batch_Y.to(config['device'])
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_time = time.time() - val_start_time
        writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch)
        writer.add_scalar("Time/val_seconds", val_time, epoch)

        scheduler.step()

        # 📊 计算并显示epoch级别的统计信息
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        elapsed_total_time = time.time() - training_start_time
        epochs_completed = epoch + 1
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = total_epochs - epochs_completed
        estimated_remaining_time = avg_epoch_time * remaining_epochs
        
        progress_percent = (epochs_completed / total_epochs) * 100
        total_elapsed = time.time() - training_start_time
        
        print(f"🟢 Epoch {epoch+1:3d}/{total_epochs} ({progress_percent:5.1f}%) | "
              f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
              f"Time: {epoch_time:.2f}s | 总用时: {format_time(total_elapsed)} | "
              f"ETA: {format_time(estimated_remaining_time)}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_tcn_lstm_model.pth')
            print(f"  💾 模型已保存 (最佳验证损失: {best_val_loss:.6f})")

    # 🎉 训练完成
    total_training_time = time.time() - training_start_time
    print(f"\n{'='*70}")
    print(f"🎉 训练完成！")
    print(f"{'='*70}")
    print(f"⏱️  总用时: {format_time(total_training_time)}")
    print(f"📊 最佳验证损失: {best_val_loss:.6f}")
    print(f"📦 最终Batch Size: {config['batch_size']}")
    print(f"📝 最终学习率: {optimizer.param_groups[0]['lr']:.6e}")
    print(f"{'='*70}\n")
    
    writer.close()

    if not config['test_mode']:
        print("📊 生成预测可视化图表...")
        visualize_predictions(model, val_loader, processor)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    train_model()
