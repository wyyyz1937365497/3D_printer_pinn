import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import matplotlib.pyplot as plt

# ==================== 配置参数 ====================
config = {
    'data_path': 'enterprise_dataset/printer_enterprise_data.csv', # 数据路径
    'seq_len': 200,          # 序列长度 (根据dt=0.5s, 200步即100秒的历史窗口)
    'batch_size': 512,       # 批次大小 (22GB显存可以开很大)
    'hidden_dim': 128,       # LSTM隐藏层维度
    'tcn_channels': [64, 64, 128], # TCN各层通道数
    'lr': 1e-3,              # 学习率
    'epochs': 50,            # 训练轮数
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,        # 数据加载线程数
}

# ==================== 1. 数据预处理模块 ====================

class PrinterDataProcessor:
    def __init__(self, data_path, seq_len):
        self.df = pd.read_csv(data_path)
        self.seq_len = seq_len
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        
        # 定义输入和输出列
        self.input_cols = ['ctrl_T_target', 'ctrl_speed_set', 'ctrl_heater_base']
        # 增加 time_step 作为辅助特征 (归一化后的时间)
        self.target_cols = ['temperature_C', 'vibration_disp_m', 'vibration_vel_m_s', 
                           'motor_current_A', 'pressure_bar', 'acoustic_signal']
        
        print(f"Loaded data: {self.df.shape}")
        self.prepare_data()

    def prepare_data(self):
        """将长格式数据转换为滑动窗口序列"""
        X_list = []
        Y_list = []
        
        # 按机器ID分组处理
        grouped = self.df.groupby('machine_id')
        
        print("Processing sequences...")
        for machine_id, group in grouped:
            group = group.sort_values('timestamp').reset_index(drop=True)
            
            # 提取原始数据
            X_raw = group[self.input_cols].values
            Y_raw = group[self.target_cols].values
            
            # 滑动窗口切分
            # 形状: (Samples, Features)
            total_len = len(group)
            if total_len < self.seq_len + 1:
                continue # 跳过数据不足的机器
                
            for i in range(total_len - self.seq_len):
                X_list.append(X_raw[i:i+self.seq_len])
                Y_list.append(Y_raw[i+self.seq_len]) # 预测下一个时间点
                
        # 转换为numpy数组
        self.X_seq = np.array(X_list, dtype=np.float32)
        self.Y_seq = np.array(Y_list, dtype=np.float32)
        
        print(f"Total sequences generated: {len(self.X_seq)}")
        
        # 归一化处理
        # 注意：这里对序列展平进行fit，也可以考虑按时间步fit
        n_samples, t_steps, n_features = self.X_seq.shape
        self.X_seq = self.X_seq.reshape(-1, n_features)
        self.scaler_X.fit(self.X_seq)
        self.X_seq = self.scaler_X.transform(self.X_seq).reshape(n_samples, t_steps, n_features)
        
        self.scaler_Y.fit(self.Y_seq)
        self.Y_seq = self.scaler_Y.transform(self.Y_seq)
        
        # 划分训练集和验证集 (80% 训练)
        split_idx = int(len(self.X_seq) * 0.8)
        self.train_X, self.val_X = self.X_seq[:split_idx], self.X_seq[split_idx:]
        self.train_Y, self.val_Y = self.Y_seq[:split_idx], self.Y_seq[split_idx:]

class PrinterDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ==================== 2. 模型定义 ====================

class TemporalBlock(nn.Module):
    """TCN 基础块: Conv1d + ReLU + Dropout + Residual"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: [Batch, Channel, Time]
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    """时间卷积网络"""
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
        # x: [Batch, Seq_Len, Features] -> [Batch, Features, Seq_Len]
        x = x.transpose(1, 2)
        out = self.network(x)
        # out: [Batch, Channels, Seq_Len] -> [Batch, Seq_Len, Channels]
        return out.transpose(1, 2)

class TCNLSTMModel(nn.Module):
    def __init__(self, input_dim, tcn_channels, hidden_dim, output_dim):
        super(TCNLSTMModel, self).__init__()
        
        # TCN 层
        self.tcn = TCN(input_dim, tcn_channels)
        tcn_output_dim = tcn_channels[-1]
        
        # LSTM 层
        self.lstm = nn.LSTM(tcn_output_dim, hidden_dim, num_layers=2, 
                           batch_first=True, dropout=0.1, bidirectional=False)
        
        # 全连接输出层
        # 我们只取最后一个时间步的输出进行预测
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x: [Batch, Seq_Len, Input_Dim]
        
        # 1. TCN 特征提取
        tcn_out = self.tcn(x) # [Batch, Seq_Len, TCN_Channels]
        
        # 2. LSTM 时序建模
        lstm_out, (h_n, c_n) = self.lstm(tcn_out) 
        
        # 3. 取最后一个时间步的隐藏状态
        last_step_out = lstm_out[:, -1, :] # [Batch, Hidden_Dim]
        
        # 4. 输出预测
        prediction = self.fc(last_step_out) # [Batch, Output_Dim]
        
        return prediction

# ==================== 3. 训练与评估 ====================

def train_model():
    # 1. 准备数据
    if not os.path.exists(config['data_path']):
        print(f"Error: Data file not found at {config['data_path']}")
        print("Please run the MATLAB script first or update the path.")
        return

    processor = PrinterDataProcessor(config['data_path'], config['seq_len'])
    
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
    
    # 多GPU支持
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
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
    
    print("Start Training...")
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(config['device']), batch_Y.to(config['device'])
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            
            loss.backward()
            # 梯度裁剪防止爆炸
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
        
        print(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_tcn_lstm_model.pth')
            print(f"  -> Model Saved (Val Loss improved to {best_val_loss:.6f})")

    # ==================== 4. 结果可视化 ====================
    print("Training Finished. Plotting results...")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss (Scaled)')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 简单预测演示
    visualize_predictions(model, val_loader, processor)

def visualize_predictions(model, loader, processor):
    model.eval()
    # 取一个batch预测
    with torch.no_grad():
        for batch_X, batch_Y in loader:
            batch_X, batch_Y = batch_X.to(config['device']), batch_Y.to(config['device'])
            preds = model(batch_X)
            break
            
    # 只画前100个点的第0个特征
    preds_np = preds.cpu().numpy()
    targets_np = batch_Y.cpu().numpy()
    
    # 反归一化
    preds_real = processor.scaler_Y.inverse_transform(preds_np)
    targets_real = processor.scaler_Y.inverse_transform(targets_np)
    
    plt.figure(figsize=(12, 8))
    for i in range(6): # 6个输出维度
        plt.subplot(3, 2, i+1)
        plt.plot(targets_real[:100, i], label='Ground Truth', alpha=0.7)
        plt.plot(preds_real[:100, i], label='Prediction', linestyle='--')
        plt.title(f'Feature: {processor.target_cols[i]}')
        plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_model()
