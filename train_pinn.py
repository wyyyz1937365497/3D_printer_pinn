import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import time
import gc
import time
import os
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# ==================== 配置参数 ====================
config = {
    # 原始 CSV 路径
    'data_path': 'enterprise_dataset/printer_enterprise_data.csv',

    # ⭐ 强烈建议把缓存目录放在高速 SSD 上（例如 D 盘或 E 盘）
    'cache_dir': './data_cache/',  # 可改为 'D:/data_cache/' 或 'E:/data_cache/'

    'seq_len': 200,
    'batch_size': 512,
    'hidden_dim': 128,
    'tcn_channels': [64, 64, 128],
    'lr': 1e-3,
    'epochs': 50,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 0,  # 使用 memmap 时建议 0
    'test_mode': False,
    'test_samples': 1000,
}

# 随机种子（保证预处理阶段的 shuffle 可复现，可选）
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)  # 推荐使用 Generator 实例<span data-allow-html class='source-item source-aggregated' data-group-key='source-group-0' data-url='https://numpy.org/doc/2.2/reference/random/generated/numpy.random.seed.html' data-id='turn0search5'><span data-allow-html class='source-item-num' data-group-key='source-group-0' data-id='turn0search5' data-url='https://numpy.org/doc/2.2/reference/random/generated/numpy.random.seed.html'><span class='source-item-num-name' data-allow-html>numpy.org</span><span data-allow-html class='source-item-num-count'>+1</span></span><span data-allow-html class='source-zhanweifu-ai-search' data-group-key='source-group-0' data-id='turn0search5' data-url='https://numpy.org/doc/2.2/reference/random/generated/numpy.random.seed.html'><div data-allow-html class='source-info' data-url='https://numpy.org/doc/2.2/reference/random/generated/numpy.random.seed.html'><div data-allow-html class='source-switcher'><span data-allow-html class='switch-counter'>1/2</span><div data-allow-html class='switch-btn-container'><img data-allow-html class='switch-btn prev dark-icon' data-group-key='source-group-0' data-direction='prev' aria-label='上一个' src='/img/arrow_down_dark.e7d9539e.svg' /><img data-allow-html class='switch-btn next dark-icon' data-group-key='source-group-0' data-direction='next' aria-label='下一个' src='/img/arrow_down_dark.e7d9539e.svg' /></div></div><div data-allow-html data-id='turn0search5' class='source-title'  data-url='https://numpy.org/doc/2.2/reference/random/generated/numpy.random.seed.html'>numpy.random.seed — NumPy v2.2 Manual</div><div data-allow-html class='source-detail source-detail-dark' data-id='turn0search5' data-url='https://numpy.org/doc/2.2/reference/random/generated/numpy.random.seed.html'><span data-allow-html data-id='turn0search5' data-url='https://numpy.org/doc/2.2/reference/random/generated/numpy.random.seed.html' class='source-text'>Best practice is to use a dedicated Generator instance rathe...</span></div><div data-allow-html class='source-info-bottom-container'><div data-allow-html class='source-icon' style="background-image: url('data:image/png;base64,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')" data-url='https://numpy.org/doc/2.2/reference/random/generated/numpy.random.seed.html' data-id='turn0search5'></div><div data-allow-html class='source-text' data-url='https://numpy.org/doc/2.2/reference/random/generated/numpy.random.seed.html'>numpy.random.seed — NumPy v2.2 Manual</div><div data-allow-html class='source-date' data-url='https://numpy.org/doc/2.2/reference/random/generated/numpy.random.seed.html'>-</div></div></div></span></span>

# ==================== 1. 预处理阶段一次性 shuffle 的数据处理器 ====================

class ShuffledOnceDataProcessor:
    """
    核心改动：
    - Pass 1：扫描所有时间窗口，计算全局均值/标准差，并记录每个样本的 (machine_id, start_idx)
    - 按 0.8 划分样本索引为 train/val
    - 对 train 索引做全局 shuffle
    - Pass 2：按照打乱后的 train 索引、原始顺序的 val 索引，分别归一化并写入 memmap
    - 训练时 DataLoader shuffle=False 即可，因为磁盘上已经是乱序数据
    """

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
        """流式处理：计算统计量 + 记录样本索引 -> 划分 train/val -> train shuffle -> 写入 memmap"""

        df = pd.read_csv(self.data_path)
        print(f"✅ 原始数据加载: {df.shape}")

        # ----------------------------------------
        # Pass 1/2：计算全局统计量，并收集所有样本索引
        # ----------------------------------------
        print("📊 [Pass 1/2] 计算全局统计量 + 收集样本索引...")
        start_time = time.time()

        X_sum = np.zeros(len(self.input_cols), dtype=np.float64)
        X_sq_sum = np.zeros(len(self.input_cols), dtype=np.float64)
        Y_sum = np.zeros(len(self.target_cols), dtype=np.float64)
        Y_sq_sum = np.zeros(len(self.target_cols), dtype=np.float64)
        count = 0

        # 样本索引列表：每个元素是 (machine_id, start_idx)
        # 这里我们保存成 (int_machine_id, int_start_idx) 的元组列表
        sample_indices = []  # List[Tuple[int, int]]

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

        # 划分数据集（按样本数量，而不是按机器）
        self.split_idx = int(self.total_samples * 0.8)
        self.train_len = self.split_idx
        self.val_len = self.total_samples - self.split_idx
        print(f"   训练集: {self.train_len}, 验证集: {self.val_len}")

        # 划分样本索引列表
        train_sample_indices = sample_indices[:self.split_idx]
        val_sample_indices = sample_indices[self.split_idx:]

        # ----------------------------------------
        # 对训练集样本索引做全局 shuffle（预处理阶段一次性付完 shuffle 成本）
        # ----------------------------------------
        print("🔀 对训练集样本索引做全局 shuffle...")
        train_sample_indices = list(train_sample_indices)  # 确保是列表
        rng.shuffle(train_sample_indices)                  # 使用 numpy Generator 的 shuffle<span data-allow-html class='source-item source-aggregated' data-group-key='source-group-1' data-url='https://numpy.org/doc/2.1/reference/random/generated/numpy.random.shuffle.html' data-id='turn0search11'><span data-allow-html class='source-item-num' data-group-key='source-group-1' data-id='turn0search11' data-url='https://numpy.org/doc/2.1/reference/random/generated/numpy.random.shuffle.html'><span class='source-item-num-name' data-allow-html>numpy.org</span><span data-allow-html class='source-item-num-count'>+1</span></span><span data-allow-html class='source-zhanweifu-ai-search' data-group-key='source-group-1' data-id='turn0search11' data-url='https://numpy.org/doc/2.1/reference/random/generated/numpy.random.shuffle.html'><div data-allow-html class='source-info' data-url='https://numpy.org/doc/2.1/reference/random/generated/numpy.random.shuffle.html'><div data-allow-html class='source-switcher'><span data-allow-html class='switch-counter'>1/2</span><div data-allow-html class='switch-btn-container'><img data-allow-html class='switch-btn prev dark-icon' data-group-key='source-group-1' data-direction='prev' aria-label='上一个' src='/img/arrow_down_dark.e7d9539e.svg' /><img data-allow-html class='switch-btn next dark-icon' data-group-key='source-group-1' data-direction='next' aria-label='下一个' src='/img/arrow_down_dark.e7d9539e.svg' /></div></div><div data-allow-html data-id='turn0search11' class='source-title'  data-url='https://numpy.org/doc/2.1/reference/random/generated/numpy.random.shuffle.html'>numpy.random.shuffle — NumPy v2.1 Manual</div><div data-allow-html class='source-detail source-detail-dark' data-id='turn0search11' data-url='https://numpy.org/doc/2.1/reference/random/generated/numpy.random.shuffle.html'><span data-allow-html data-id='turn0search11' data-url='https://numpy.org/doc/2.1/reference/random/generated/numpy.random.shuffle.html' class='source-text'>Modify a sequence in-place by shuffling its contents. This f...</span></div><div data-allow-html class='source-info-bottom-container'><div data-allow-html class='source-icon' style="background-image: url('data:image/png;base64,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')" data-url='https://numpy.org/doc/2.1/reference/random/generated/numpy.random.shuffle.html' data-id='turn0search11'></div><div data-allow-html class='source-text' data-url='https://numpy.org/doc/2.1/reference/random/generated/numpy.random.shuffle.html'>numpy.random.shuffle — NumPy v2.1 Manual</div><div data-allow-html class='source-date' data-url='https://numpy.org/doc/2.1/reference/random/generated/numpy.random.shuffle.html'>-</div></div></div></span></span>

        # 保存归一化参数到磁盘
        print("💾 保存归一化参数")
        scaler_path = os.path.join(self.cache_dir, 'scaler_stats.npz')
        np.savez(scaler_path,
                 mean_X=self.mean_X, std_X=self.std_X,
                 mean_Y=self.mean_Y, std_Y=self.std_Y)
        print(f"   已保存至: {scaler_path}")

        # 可选：保存样本索引（方便后面检查、复现）
        indices_path = os.path.join(self.cache_dir, 'sample_indices.npz')
        np.savez(indices_path,
                 train_sample_indices=np.array(train_sample_indices, dtype=object),
                 val_sample_indices=np.array(val_sample_indices, dtype=object))
        print(f"   样本索引已保存至: {indices_path}")

        # ----------------------------------------
        # Pass 2/2：按照 train/val 的索引顺序，归一化并写入 memmap
        # ----------------------------------------
        print("💾 [Pass 2/2] 按 train/val 索引顺序写入 memmap 缓存文件 (这可能需要几分钟)...")

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

        # 辅助函数：从原始 df 中根据 (machine_id, start_idx) 取窗口并归一化
        def write_samples(sample_indices_list, is_train):
            nonlocal train_ptr, val_ptr

            # 为了避免重复对 df 做大 groupby，我们在这里先按 machine_id 再遍历
            # 一种简单的方式：先按 machine_id 把 group 缓存起来（如果机器数量不太多）
            # 这里为了内存安全，我们直接对 sample_indices 列表按 machine_id 排序后遍历
            # 相同 machine_id 的样本连续，减少 groupby 的次数

            # 先按 machine_id 排序（保持时间顺序）
            sorted_indices = sorted(sample_indices_list, key=lambda x: x[0])

            current_machine_id = None
            group_X = None
            group_Y = None

            for (mid, start) in sorted_indices:
                if mid != current_machine_id:
                    # 换了一个新机器，重新获取该机器的数据
                    sub_df = df[df['machine_id'] == mid].sort_values('timestamp').reset_index(drop=True)
                    group_X = sub_df[self.input_cols].values
                    group_Y = sub_df[self.target_cols].values
                    current_machine_id = mid

                # 取窗口
                x_win = group_X[start:start + self.seq_len]
                y_win = group_Y[start + self.seq_len]

                # 归一化
                x_norm = (x_win - self.mean_X) / self.std_X
                y_norm = (y_win - self.mean_Y) / self.std_Y

                # 写入 memmap
                if is_train:
                    self.train_X[train_ptr] = x_norm.astype(np.float32)
                    self.train_Y[train_ptr] = y_norm.astype(np.float32)
                    train_ptr += 1
                else:
                    self.val_X[val_ptr] = x_norm.astype(np.float32)
                    self.val_Y[val_ptr] = y_norm.astype(np.float32)
                    val_ptr += 1

        # 写入训练集（注意：train_sample_indices 已经是打乱过的顺序）
        write_samples(train_sample_indices, is_train=True)

        # 写入验证集（原始顺序）
        write_samples(val_sample_indices, is_train=False)

        print("✅ 缓存写入完成！")

        # 关闭并清理 memmap 对象
        del self.train_X, self.train_Y, self.val_X, self.val_Y
        gc.collect()

        # 重新以只读方式加载，供训练使用
        self.load_metadata()

    def load_metadata(self):
        """加载已缓存的 Memmap 和 Scaler"""

        scaler_path = os.path.join(self.cache_dir, 'scaler_stats.npz')
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"找不到 Scaler 文件: {scaler_path}，请重新生成缓存。")

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
        """将归一化的预测值还原为真实物理值 (用于可视化)"""
        return y_norm * self.std_Y + self.mean_Y


class MMapDataset(Dataset):
    def __init__(self, X_mmap, Y_mmap):
        self.X = X_mmap
        self.Y = Y_mmap

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx].copy(), self.Y[idx].copy()


# ==================== 2. 模型定义（不变） ====================

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


# ==================== 3. 训练与可视化（几乎不变） ====================

def visualize_predictions(model, loader, processor):
    """使用真实的物理单位进行可视化"""
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
        # 只画前 100 个点
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
    print("=" * 60)
    print("🚀 TCN-LSTM 训练 (预处理阶段一次性 shuffle)")
    print("=" * 60)

    # 使用新的 ShuffledOnceDataProcessor
    processor = ShuffledOnceDataProcessor(
        config['data_path'],
        config['seq_len'],
        config['cache_dir'],
        test_mode=config['test_mode'],
        test_samples=config['test_samples']
    )

    train_dataset = MMapDataset(processor.train_X, processor.train_Y)
    val_dataset = MMapDataset(processor.val_X, processor.val_Y)

    # 关键改动：这里 DataLoader 的 shuffle 设置为 False
    # 因为磁盘上 train 数据已经是乱序了，不需要再在 DataLoader 里 shuffle
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=False,    # ⭐ 这里一定要 False
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.MSELoss()

    print("🏋️ 开始训练...")
    best_val_loss = float('inf')
    print_every = 100   # 每 100 个 batch 打印一次，方便观察进度

    for epoch in range(config['epochs']):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0

        for batch_idx, (batch_X, batch_Y) in enumerate(train_loader):
            batch_X, batch_Y = batch_X.to(config['device']), batch_Y.to(config['device'])

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % print_every == 0:
                avg_so_far = epoch_loss / (batch_idx + 1)
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Avg Train Loss: {avg_so_far:.6f}")

        avg_train_loss = epoch_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                batch_X, batch_Y = batch_X.to(config['device']), batch_Y.to(config['device'])
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Time: {epoch_time:.2f}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_tcn_lstm_model.pth')
            print(f"  💾 模型已保存")

    # 训练结束后可视化
    if not config['test_mode']:
        print("\n生成预测可视化图表...")
        visualize_predictions(model, val_loader, processor)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    train_model()
