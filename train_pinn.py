import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
import time
import gc

# ==================== 配置参数 ====================
config = {
    'data_path': 'enterprise_dataset/printer_enterprise_data.csv',  # 原始 CSV 路径
    
    # ⭐ 修改这里：将缓存目录设置在你的高速 SSD 上 (例如 D盘 或 E盘)
    # 这将大幅提升 DataLoader 的读取速度
    'cache_dir': './data_cache/',  
    
    'seq_len': 200,
    'batch_size': 512,
    'hidden_dim': 128,
    'tcn_channels': [64, 64, 128],
    'lr': 1e-3,
    'epochs': 50,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 0, # 使用 memmap 时建议 0
    'test_mode': False,
    'test_samples': 1000,
}

# ==================== 1. 最终版 数据处理器 ====================

class EfficientDataProcessor:
    def __init__(self, data_path, seq_len, cache_dir, test_mode=False, test_samples=1000):
        self.data_path = data_path
        self.seq_len = seq_len
        self.cache_dir = cache_dir
        self.test_mode = test_mode
        self.test_samples = test_samples
        
        self.input_cols = ['ctrl_T_target', 'ctrl_speed_set', 'ctrl_heater_base']
        self.target_cols = ['temperature_C', 'vibration_disp_m', 'vibration_vel_m_s', 
                           'motor_current_A', 'pressure_bar', 'acoustic_signal']
        
        # 检查缓存是否存在
        if os.path.exists(cache_dir) and os.listdir(cache_dir):
            print(f"📦 发现缓存目录: {cache_dir}")
            self.load_metadata()
        else:
            print(f"🔄 缓存不存在，开始处理数据...")
            print(f"🚀 缓存将写入: {cache_dir}")
            os.makedirs(cache_dir, exist_ok=True)
            self.process_and_save()

    def process_and_save(self):
        """流式处理：计算统计量，归一化，并写入 mmap"""
        df = pd.read_csv(self.data_path)
        print(f"✅ 原始数据加载: {df.shape}")
        
        # 1. 第一遍扫描：计算全局统计量
        print("📊 [Pass 1/2] 计算全局统计量...")
        start_time = time.time()
        
        X_sum = np.zeros(len(self.input_cols), dtype=np.float64)
        X_sq_sum = np.zeros(len(self.input_cols), dtype=np.float64)
        Y_sum = np.zeros(len(self.target_cols), dtype=np.float64)
        Y_sq_sum = np.zeros(len(self.target_cols), dtype=np.float64)
        count = 0
        
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
                x_win = X_raw[i:i+self.seq_len]  # 移除reshape(-1)，保持二维形状(seq_len, n_features)
                y_win = Y_raw[i+self.seq_len]
                
                # 计算每个特征的统计量，而不是整个窗口的统计量
                X_sum += x_win.mean(axis=0)  # 对序列维度求平均，保留特征维度
                X_sq_sum += (x_win**2).mean(axis=0)  # 对序列维度求平均，保留特征维度
                Y_sum += y_win
                Y_sq_sum += y_win**2
                count += 1
                
                if self.test_mode and count >= self.test_samples:
                    break
            
            if self.test_mode and count >= self.test_samples:
                break
                
        # 计算均值和标准差
        self.mean_X = X_sum / count
        self.var_X = (X_sq_sum / count) - (self.mean_X ** 2)
        self.std_X = np.sqrt(self.var_X)
        
        self.mean_Y = Y_sum / count
        self.var_Y = (Y_sq_sum / count) - (self.mean_Y ** 2)
        self.std_Y = np.sqrt(self.var_Y)
        
        self.total_samples = count
        print(f"   样本总数: {self.total_samples}")
        print(f"   耗时: {time.time() - start_time:.2f}s")
        
        # 划分数据集
        self.split_idx = int(self.total_samples * 0.8)
        self.train_len = self.split_idx
        self.val_len = self.total_samples - self.split_idx
        
        print(f"   训练集: {self.train_len}, 验证集: {self.val_len}")
        
        # ⭐ 关键步骤：保存 Scaler 统计量到磁盘
        print("💾 保存归一化参数")
        scaler_path = os.path.join(self.cache_dir, 'scaler_stats.npz')
        np.savez(scaler_path,
                 mean_X=self.mean_X, std_X=self.std_X,
                 mean_Y=self.mean_Y, std_Y=self.std_Y)
        print(f"   已保存至: {scaler_path}")

        # 2. 第二遍扫描：归一化并写入 Memmap
        print("💾 [Pass 2/2] 写入 mmap 缓存文件 (这可能需要几分钟)...")
        
        # 准备文件路径
        mmap_files = {
            'train_X': os.path.join(self.cache_dir, 'train_X.npy'),
            'train_Y': os.path.join(self.cache_dir, 'train_Y.npy'),
            'val_X': os.path.join(self.cache_dir, 'val_X.npy'),
            'val_Y': os.path.join(self.cache_dir, 'val_Y.npy'),
        }
        
        # 创建并初始化 Memmap 文件 (w+ 模式会创建并覆盖)
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
        
        # 写入指针
        train_ptr = 0
        val_ptr = 0
        current_idx = 0
        
        for machine_id, group in grouped:
            group = group.sort_values('timestamp').reset_index(drop=True)
            X_raw = group[self.input_cols].values
            Y_raw = group[self.target_cols].values
            
            total_len = len(group)
            if total_len < self.seq_len + 1:
                continue
                
            n_windows = total_len - self.seq_len
            
            for i in range(n_windows):
                if self.test_mode and current_idx >= self.test_samples:
                    break
                
                # 归一化 (X)
                x_win = X_raw[i:i+self.seq_len]
                x_norm = (x_win - self.mean_X) / self.std_X
                
                # 归一化 (Y)
                y_win = Y_raw[i+self.seq_len]
                y_norm = (y_win - self.mean_Y) / self.std_Y
                
                # 写入 Memmap
                if current_idx < self.train_len:
                    self.train_X[train_ptr] = x_norm.astype(np.float32)
                    self.train_Y[train_ptr] = y_norm.astype(np.float32)
                    train_ptr += 1
                else:
                    self.val_X[val_ptr] = x_norm.astype(np.float32)
                    self.val_Y[val_ptr] = y_norm.astype(np.float32)
                    val_ptr += 1
                    
                current_idx += 1
            
            if self.test_mode and current_idx >= self.test_samples:
                break
        
        print("✅ 缓存写入完成！")
        
        # 不删除mmap对象，而是关闭并重新以只读模式加载
        # 为了确保在训练过程中可以访问这些属性，我们需要重新加载它们
        # 删除这些临时属性，然后在load_metadata中重新加载
        del self.train_X, self.train_Y, self.val_X, self.val_Y
        gc.collect()
        
        # 重新加载数据以供训练使用
        self.load_metadata()

    def load_metadata(self):
        """加载已缓存的 Memmap 和 Scaler"""
        # 1. 加载 Scaler 参数
        scaler_path = os.path.join(self.cache_dir, 'scaler_stats.npz')
        if not os.path.exists(scaler_path):
             raise FileNotFoundError(f"找不到 Scaler 文件: {scaler_path}，请重新生成缓存。")
        
        data = np.load(scaler_path)
        self.mean_X = data['mean_X']
        self.std_X = data['std_X']
        self.mean_Y = data['mean_Y']
        self.std_Y = data['std_Y']
        print("✅ 归一化参数加载成功")
        
        # 2. 加载 Memmap 数据
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
        return self.X[idx], self.Y[idx]

# ==================== 2. 模型定义 ====================

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
            in_channels = num_inputs if i == 0 else num_channels[i-1]
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

# ==================== 3. 训练与可视化 ====================

def visualize_predictions(model, loader, processor):
    """使用真实的物理单位进行可视化"""
    model.eval()
    with torch.no_grad():
        for batch_X, batch_Y in loader:
            batch_X, batch_Y = batch_X.to(config['device']), batch_Y.to(config['device'])
            preds = model(batch_X)
            break
            
    # 移回 CPU 并转为 numpy
    preds_np = preds.cpu().numpy()
    targets_np = batch_Y.cpu().numpy()
    
    # ⭐ 使用 processor 的方法进行反归一化
    preds_real = processor.inverse_transform_y(preds_np)
    targets_real = processor.inverse_transform_y(targets_np)
    
    plt.figure(figsize=(12, 8))
    for i in range(6):
        plt.subplot(3, 2, i+1)
        # 只画前100个点
        plt.plot(targets_real[:100, i], label='Ground Truth', alpha=0.7)
        plt.plot(preds_real[:100, i], label='Prediction', linestyle='--')
        plt.title(f'Feature: {processor.target_cols[i]}')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()

def train_model():
    print("="*60)
    print("🚀 最终版 TCN-LSTM 训练 (支持 Scaler 存取)")
    print("="*60)
    
    processor = EfficientDataProcessor(
        config['data_path'], 
        config['seq_len'],
        config['cache_dir'], # 这里使用你在 config 中设置的高速磁盘路径
        test_mode=config['test_mode'],
        test_samples=config['test_samples']
    )
    
    train_dataset = MMapDataset(processor.train_X, processor.train_Y)
    val_dataset = MMapDataset(processor.val_X, processor.val_Y)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=0, pin_memory=True)
    
    # 初始化模型
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
