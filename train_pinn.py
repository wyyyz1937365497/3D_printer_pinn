import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
import time

# ==================== 配置参数 ====================
config = {
    'data_path': 'enterprise_dataset/printer_enterprise_data.csv',
    'cache_path': 'enterprise_dataset/processed_data.pt',  # 数据缓存路径
    'seq_len': 200,          # 序列长度
    'batch_size': 512,       # 批次大小
    'hidden_dim': 128,       # LSTM隐藏层维度
    'tcn_channels': [64, 64, 128], # TCN各层通道数
    'lr': 1e-3,
    'epochs': 50,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
    'test_mode': True,      # ⭐ 测试模式：只加载少量数据快速验证
    'test_samples': 1000,    # 测试模式使用的样本数
    'use_cache': True,      # 是否使用缓存
}

# ==================== 1. 数据预处理模块 ====================

class PrinterDataProcessor:
    def __init__(self, data_path, seq_len, use_cache=True, test_mode=False, test_samples=1000):
        self.data_path = data_path
        self.seq_len = seq_len
        self.use_cache = use_cache
        self.test_mode = test_mode
        self.test_samples = test_samples
        
        # 定义输入和输出列
        self.input_cols = ['ctrl_T_target', 'ctrl_speed_set', 'ctrl_heater_base']
        self.target_cols = ['temperature_C', 'vibration_disp_m', 'vibration_vel_m_s', 
                           'motor_current_A', 'pressure_bar', 'acoustic_signal']
        
        # 检查缓存
        cache_file = 'enterprise_dataset/processed_data.pt'
        if use_cache and os.path.exists(cache_file):
            print(f"📦 发现缓存文件 {cache_file}，直接加载...")
            self.load_from_cache(cache_file)
        else:
            print(f"🔄 缓存不存在或禁用，开始处理原始数据...")
            self.prepare_data()
            if use_cache:
                self.save_to_cache(cache_file)
    
    def prepare_data(self):
        """处理原始数据并分割"""
        print(f"📂 加载数据: {self.data_path}")
        df = pd.read_csv(self.data_path)
        print(f"✅ 原始数据形状: {df.shape}")
        
        # =============================================================
        # 🔧 修复核心: 处理 NaN 和 Inf 值
        # =============================================================
        print("🔍 检查异常值")
        
        # 1. 统计异常值
        nan_counts = df.isna().sum().sum()
        # 检查无穷大
        inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        
        print(f"   发现 NaN: {nan_counts}, Inf: {inf_counts}")
        
        # 2. 将无穷大替换为 NaN (以便统一处理)
        # replace inf with nan
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # 3. 使用新版语法进行填充 (解决 FutureWarning)
        # ffill: forward fill, bfill: backward fill
        if df.isna().sum().sum() > 0:
            print("⚠️ 正在清理异常值并插值...")
            df = df.ffill().bfill()
            
            # 再次检查：如果整列都是异常值，直接删除该列（不太可能，但为了健壮性）
            df.dropna(axis=1, how='all', inplace=True)
            print("✅ 异常值处理完成")
        # =============================================================
        
        # 按机器ID分组处理
        X_list = []
        Y_list = []
        
        grouped = df.groupby('machine_id')
        print(f"🔄 处理 {len(grouped)} 台机器的数据...")
        
        start_time = time.time()
        for idx, (machine_id, group) in enumerate(grouped):
            if idx % 10 == 0:
                print(f"   进度: {idx}/{len(grouped)} 机器")
            
            group = group.sort_values('timestamp').reset_index(drop=True)
            
            # 提取数据
            X_raw = group[self.input_cols].values
            Y_raw = group[self.target_cols].values
            
            # 滑动窗口
            total_len = len(group)
            if total_len < self.seq_len + 1:
                continue
                
            # 这里增加一个安全检查：如果 slice 中还有 NaN (虽然前面处理过)，跳过
            if np.isnan(X_raw).any() or np.isnan(Y_raw).any():
                continue
                
            for i in range(total_len - self.seq_len):
                X_list.append(X_raw[i:i+self.seq_len])
                Y_list.append(Y_raw[i+self.seq_len])
        
        self.X_seq = np.array(X_list, dtype=np.float32)
        self.Y_seq = np.array(Y_list, dtype=np.float32)
        
        elapsed = time.time() - start_time
        print(f"✅ 序列生成完成: {len(self.X_seq)} 个序列，耗时 {elapsed:.2f}s")
        
        if len(self.X_seq) == 0:
            raise ValueError("没有生成有效序列！请检查数据是否全部为空。")
        
        # 测试模式：只取少量数据
        if self.test_mode:
            print(f"🧪 测试模式：只使用前 {self.test_samples} 个样本")
            self.X_seq = self.X_seq[:self.test_samples]
            self.Y_seq = self.Y_seq[:self.test_samples]
        
        # 归一化
        print("📊 开始归一化...")
        n_samples, t_steps, n_features = self.X_seq.shape
        X_flat = self.X_seq.reshape(-1, n_features)
        
        self.scaler_X = StandardScaler()
        self.X_seq = self.scaler_X.fit_transform(X_flat).reshape(n_samples, t_steps, n_features)
        
        self.scaler_Y = StandardScaler()
        self.Y_seq = self.scaler_Y.fit_transform(self.Y_seq)
        
        # 划分数据集
        split_idx = int(len(self.X_seq) * 0.8)
        self.train_X = self.X_seq[:split_idx]
        self.train_Y = self.Y_seq[:split_idx]
        self.val_X = self.X_seq[split_idx:]
        self.val_Y = self.Y_seq[split_idx:]
        
        print(f"✅ 数据集划分完成:")
        print(f"   训练集: {len(self.train_X)} 样本")
        print(f"   验证集: {len(self.val_X)} 样本")
    
    def save_to_cache(self, cache_path):
        """保存处理后的数据到缓存"""
        print(f"💾 保存缓存到 {cache_path}...")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        cache_data = {
            'train_X': self.train_X,
            'train_Y': self.train_Y,
            'val_X': self.val_X,
            'val_Y': self.val_Y,
            'scaler_X_mean': self.scaler_X.mean_,
            'scaler_X_scale': self.scaler_X.scale_,
            'scaler_Y_mean': self.scaler_Y.mean_,
            'scaler_Y_scale': self.scaler_Y.scale_,
            'input_cols': self.input_cols,
            'target_cols': self.target_cols,
        }
        
        torch.save(cache_data, cache_path)
        file_size = os.path.getsize(cache_path) / 1024 / 1024  # MB
        print(f"✅ 缓存保存成功! 文件大小: {file_size:.2f} MB")
    
    def load_from_cache(self, cache_path):
        """从缓存加载数据"""
        cache_data = torch.load(cache_path)
        
        self.train_X = cache_data['train_X']
        self.train_Y = cache_data['train_Y']
        self.val_X = cache_data['val_X']
        self.val_Y = cache_data['val_Y']
        
        # 重建scaler
        self.scaler_X = StandardScaler()
        self.scaler_X.mean_ = cache_data['scaler_X_mean']
        self.scaler_X.scale_ = cache_data['scaler_X_scale']
        
        self.scaler_Y = StandardScaler()
        self.scaler_Y.mean_ = cache_data['scaler_Y_mean']
        self.scaler_Y.scale_ = cache_data['scaler_Y_scale']
        
        self.input_cols = cache_data['input_cols']
        self.target_cols = cache_data['target_cols']
        
        print(f"✅ 缓存加载成功!")
        print(f"   训练集: {len(self.train_X)} 样本")
        print(f"   验证集: {len(self.val_X)} 样本")

class PrinterDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ==================== 2. 修复后的 TCN 模型 ====================

class TemporalBlock(nn.Module):
    """修复后的 TCN 基础块"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # 使用正确的 padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # 1x1 卷积用于匹配维度
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.kernel_size = kernel_size
        self.dilation = dilation
        
    def forward(self, x):
        # x shape: [Batch, Channel, Time]
        out = self.net(x)
        
        # 移除padding以保持因果性
        # padding = (kernel_size - 1) * dilation
        # 我们需要移除最后的padding个时间步
        pad = (self.kernel_size - 1) * self.dilation
        out = out[:, :, :-pad]
        
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    """修复后的时间卷积网络"""
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size  # 因果卷积的padding
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                   dilation=dilation_size, padding=padding, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        # x: [Batch, Seq_Len, Features] -> [Batch, Features, Seq_Len]
        x = x.transpose(1, 2)
        out = self.network(x)
        # out: [Batch, Channels, Seq_Len] -> [Batch, Seq_Len, Channels]
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
        # x: [Batch, Seq_Len, Input_Dim]
        tcn_out = self.tcn(x)  # [Batch, Seq_Len, TCN_Channels]
        lstm_out, (h_n, c_n) = self.lstm(tcn_out) 
        last_step_out = lstm_out[:, -1, :]  # [Batch, Hidden_Dim]
        prediction = self.fc(last_step_out)  # [Batch, Output_Dim]
        
        return prediction

# ==================== 3. 训练与评估 ====================

def train_model():
    # 打印配置信息
    print("="*60)
    print("🚀 TCN-LSTM 训练开始")
    print("="*60)
    print(f"测试模式: {'✅ 启用 (快速验证)' if config['test_mode'] else '❌ 禁用 (完整训练)'}")
    print(f"使用缓存: {'✅ 启用' if config['use_cache'] else '❌ 禁用'}")
    print(f"设备: {config['device']}")
    print(f"批次大小: {config['batch_size']}")
    print("="*60)
    print()
    
    # 1. 准备数据
    if not os.path.exists(config['data_path']):
        print(f"❌ 错误: 数据文件不存在 {config['data_path']}")
        print("请先运行 MATLAB 脚本生成数据")
        return

    processor = PrinterDataProcessor(
        config['data_path'], 
        config['seq_len'],
        use_cache=config['use_cache'],
        test_mode=config['test_mode'],
        test_samples=config['test_samples']
    )
    
    train_dataset = PrinterDataset(processor.train_X, processor.train_Y)
    val_dataset = PrinterDataset(processor.val_X, processor.val_Y)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=config['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=config['num_workers'], pin_memory=True)
    
    # 2. 初始化模型
    input_dim = len(processor.input_cols)
    output_dim = len(processor.target_cols)
    
    model = TCNLSTMModel(input_dim, config['tcn_channels'], config['hidden_dim'], output_dim)
    
    if torch.cuda.device_count() > 1:
        print(f"🎮 使用 {torch.cuda.device_count()} 个 GPU!")
        model = nn.DataParallel(model)
        
    model = model.to(config['device'])
    
    # 3. 优化器与损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.MSELoss()
    
    # 4. 训练循环
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("🏋️ 开始训练...")
    print("="*60)
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        
        # 训练
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
        
        # 验证
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
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1:3d}/{config['epochs']} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | "
              f"Time: {epoch_time:.2f}s")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_tcn_lstm_model.pth')
            print(f"  💾 模型已保存 (最佳 Val Loss: {best_val_loss:.6f})")
    
    print("="*60)
    print("✅ 训练完成!")
    print("="*60)
    
    # 结果可视化
    if not config['test_mode']:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss (Scaled)')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        visualize_predictions(model, val_loader, processor)

def visualize_predictions(model, loader, processor):
    model.eval()
    with torch.no_grad():
        for batch_X, batch_Y in loader:
            batch_X, batch_Y = batch_X.to(config['device']), batch_Y.to(config['device'])
            preds = model(batch_X)
            break
            
    preds_np = preds.cpu().numpy()
    targets_np = batch_Y.cpu().numpy()
    
    # 反归一化
    preds_real = processor.scaler_Y.inverse_transform(preds_np)
    targets_real = processor.scaler_Y.inverse_transform(targets_np)
    
    plt.figure(figsize=(12, 8))
    for i in range(6):
        plt.subplot(3, 2, i+1)
        plt.plot(targets_real[:100, i], label='Ground Truth', alpha=0.7)
        plt.plot(preds_real[:100, i], label='Prediction', linestyle='--')
        plt.title(f'Feature: {processor.target_cols[i]}')
        plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_model()
