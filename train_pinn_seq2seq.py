# train_pinn_seq2seq.py (完整改进版)
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import time
import gc
import argparse
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import pickle
import matplotlib.pyplot as plt
import signal
import atexit

# ==================== 配置参数 ====================
class Config:
    def __init__(self):
        self.data_path = 'enterprise_dataset/printer_enterprise_data.csv'
        self.seq_len = 200          # 历史长度
        self.pred_len = 50          # 预测长度
        self.batch_size = 256
        self.gradient_accumulation_steps = 4
        self.model_dim = 256
        self.num_heads = 8
        self.num_layers = 6
        self.dim_feedforward = 1024
        self.dropout = 0.1
        self.lr = 2e-4
        self.epochs = 30
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_workers = 4
        self.max_samples = 200000
        self.lambda_physics = 0.05
        self.warmup_epochs = 5
        self.checkpoint_dir = './checkpoints_seq2seq'
        self.resume_from = None
        self.save_on_exit = True
        self.save_interval = 5
        self.start_epoch = 0
        self.load_optimizer_state = True  # 优化器状态加载控制
        self.original_batch_size = None   # 原始batch size（用于学习率缩放）
        
        # 列定义
        self.ctrl_cols = ['ctrl_T_target', 'ctrl_speed_set', 'ctrl_heater_base']
        self.state_cols = ['temperature_C', 'vibration_disp_m', 'vibration_vel_m_s',
                          'motor_current_A', 'pressure_bar', 'acoustic_signal']
        
        # 维度定义
        self.input_dim = len(self.ctrl_cols) + len(self.state_cols)
        self.output_dim = len(self.state_cols)
        self.ctrl_dim = len(self.ctrl_cols)

# ==================== 位置编码 ====================
class PositionalEncoding(nn.Module):
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

# ==================== Seq2Seq 模型 ====================
class PrinterPINN_Seq2Seq(nn.Module):
    def __init__(self, config):
        super(PrinterPINN_Seq2Seq, self).__init__()
        
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.ctrl_dim = config.ctrl_dim
        self.d_model = config.model_dim
        self.pred_len = config.pred_len
        
        # Encoder
        self.encoder_embedding = nn.Linear(self.input_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # Decoder
        self.decoder_embedding = nn.Linear(self.ctrl_dim, self.d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)
        
        # Output
        self.fc_out = nn.Linear(self.d_model, self.output_dim)

    def forward(self, src, tgt_ctrl):
        # Encoder
        src_emb = self.encoder_embedding(src)
        src_emb = self.pos_encoder(src_emb)
        memory = self.encoder(src_emb)
        
        # Decoder
        tgt_emb = self.decoder_embedding(tgt_ctrl)
        tgt_emb = self.pos_encoder(tgt_emb)
        output = self.decoder(tgt_emb, memory)
        
        prediction = self.fc_out(output)
        return prediction

    def physics_loss(self, y_pred, y_true):
        """物理约束损失（针对序列）"""
        loss = 0.0
        batch_size, seq_len, _ = y_pred.shape
        
        # 热传导损失：温度变化平滑
        temp_pred = y_pred[:, :, 0]
        dT_pred = torch.diff(temp_pred, dim=1)
        d2T_pred = torch.diff(dT_pred, dim=1)
        loss += torch.mean(d2T_pred ** 2)
        
        # 振动能量守恒
        disp_pred = y_pred[:, :, 1]
        vel_pred = y_pred[:, :, 2]
        dt = 1.0
        vel_from_disp = torch.diff(disp_pred, dim=1) / dt
        loss += torch.mean((vel_from_disp - vel_pred[:, :-1]) ** 2)
        
        return loss

# ==================== 数据处理器 ====================
class MemoryDataProcessor:
    def __init__(self, data_path, seq_len, pred_len, max_samples, config):
        self.data_path = data_path
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.max_samples = max_samples
        self.config = config
        
        self.input_dim = len(self.config.ctrl_cols) + len(self.config.state_cols)
        self.output_dim = len(self.config.state_cols)
        self.ctrl_dim = len(self.config.ctrl_cols)
        
        print(f"🔄 开始处理数据...")
        print(f"📊 历史长度: {seq_len}, 预测长度: {pred_len}")
        self.process_data()

    def process_data(self):
        """处理数据用于Seq2Seq训练"""
        df = pd.read_csv(self.data_path)
        print(f"✅ 原始数据加载: {df.shape}")
        
        numeric_cols = self.config.ctrl_cols + self.config.state_cols + ['fault_label']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(np.float32)
        
        all_cols = self.config.ctrl_cols + self.config.state_cols
        
        grouped = df.groupby('machine_id')
        samples = []
        count = 0
        
        print("📊 收集样本索引...")
        for machine_id, group in grouped:
            group = group.sort_values('timestamp').reset_index(drop=True)
            data_array = group[all_cols].values
            ctrl_array = group[self.config.ctrl_cols].values
            state_array = group[self.config.state_cols].values
            fault_array = group['fault_label'].values if 'fault_label' in group.columns else np.zeros(len(group))
            
            total_len = len(group)
            required_len = self.seq_len + self.pred_len
            
            if total_len < required_len:
                continue
            
            n_windows = total_len - required_len + 1
            
            for i in range(n_windows):
                if count >= self.max_samples:
                    break
                
                window_fault = fault_array[i:i+required_len]
                if np.any(window_fault == 1):
                    continue
                
                x_hist = data_array[i:i+self.seq_len]
                x_future_ctrl = ctrl_array[i+self.seq_len:i+required_len]
                y_future_state = state_array[i+self.seq_len:i+required_len]
                
                samples.append((x_hist, x_future_ctrl, y_future_state))
                count += 1
            
            if count >= self.max_samples:
                break
        
        self.total_samples = len(samples)
        self.split_idx = int(self.total_samples * 0.8)
        
        train_samples = samples[:self.split_idx]
        val_samples = samples[self.split_idx:]
        
        print(f"📊 总样本数: {self.total_samples}")
        print(f"   训练集: {len(train_samples)}, 验证集: {len(val_samples)}")
        
        print("📊 计算统计量...")
        all_x_hist = np.array([s[0] for s in train_samples])
        all_y_future = np.array([s[2] for s in train_samples])
        
        self.mean_X = all_x_hist.mean(axis=(0, 1))
        self.std_X = all_x_hist.std(axis=(0, 1))
        self.mean_Y = all_y_future.mean(axis=(0, 1))
        self.std_Y = all_y_future.std(axis=(0, 1))
        
        self.std_X[self.std_X < 1e-8] = 1.0
        self.std_Y[self.std_Y < 1e-8] = 1.0
        
        print(f"   Input Mean: {self.mean_X}")
        print(f"   Input Std: {self.std_X}")
        print(f"   Output Mean: {self.mean_Y}")
        print(f"   Output Std: {self.std_Y}")
        
        self.train_X = np.zeros((len(train_samples), self.seq_len, self.input_dim), dtype=np.float32)
        self.train_ctrl = np.zeros((len(train_samples), self.pred_len, self.ctrl_dim), dtype=np.float32)
        self.train_Y = np.zeros((len(train_samples), self.pred_len, self.output_dim), dtype=np.float32)
        
        for idx, (x_hist, x_ctrl, y_state) in enumerate(train_samples):
            self.train_X[idx] = (x_hist - self.mean_X) / self.std_X
            self.train_ctrl[idx] = (x_ctrl - self.mean_X[:self.ctrl_dim]) / self.std_X[:self.ctrl_dim]
            self.train_Y[idx] = (y_state - self.mean_Y) / self.std_Y
        
        self.val_X = np.zeros((len(val_samples), self.seq_len, self.input_dim), dtype=np.float32)
        self.val_ctrl = np.zeros((len(val_samples), self.pred_len, self.ctrl_dim), dtype=np.float32)
        self.val_Y = np.zeros((len(val_samples), self.pred_len, self.output_dim), dtype=np.float32)
        
        for idx, (x_hist, x_ctrl, y_state) in enumerate(val_samples):
            self.val_X[idx] = (x_hist - self.mean_X) / self.std_X
            self.val_ctrl[idx] = (x_ctrl - self.mean_X[:self.ctrl_dim]) / self.std_X[:self.ctrl_dim]
            self.val_Y[idx] = (y_state - self.mean_Y) / self.std_Y
        
        print(f"✅ 数据处理完成！")

    def inverse_transform_y(self, y_norm):
        return y_norm * self.std_Y + self.mean_Y

# ==================== 数据集类 ====================
class Seq2SeqDataset(Dataset):
    def __init__(self, X_hist, X_ctrl, Y):
        self.X_hist = torch.from_numpy(X_hist)
        self.X_ctrl = torch.from_numpy(X_ctrl)
        self.Y = torch.from_numpy(Y)

    def __len__(self):
        return self.X_hist.shape[0]

    def __getitem__(self, idx):
        return self.X_hist[idx], self.X_ctrl[idx], self.Y[idx]

def seq2seq_collate_fn(batch):
    x_hist, x_ctrl, y = zip(*batch)
    import torch
    # 确保所有数据都是tensor类型
    x_hist = torch.stack([torch.from_numpy(x) if isinstance(x, np.ndarray) else x for x in x_hist])
    x_ctrl = torch.stack([torch.from_numpy(x) if isinstance(x, np.ndarray) else x for x in x_ctrl])
    y = torch.stack([torch.from_numpy(y) if isinstance(y, np.ndarray) else y for y in y])
    return x_hist, x_ctrl, y

# ==================== 训练状态管理器 ====================
class TrainingStateManager:
    """管理训练状态，用于优雅退出和恢复"""
    def __init__(self, config, model, optimizer, scheduler, checkpoint_dir):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.current_epoch = 0
        self.train_loss = 0.0
        self.val_loss = 0.0
        self.best_val_loss = float('inf')
        self.training_start_time = time.time()
        self.last_epoch_time = time.time()
        
    def update_epoch(self, epoch, train_loss, val_loss):
        self.current_epoch = epoch
        self.train_loss = train_loss
        self.val_loss = val_loss
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            
        self.last_epoch_time = time.time()
    
    def save_checkpoint(self, filename):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__,
            'training_start_time': self.training_start_time
        }
        torch.save(checkpoint, filename)
        print(f"💾 检查点已保存: {filename}")
    
    def get_time_info(self):
        """获取时间信息字典"""
        elapsed_total = time.time() - self.training_start_time
        epochs_completed = self.current_epoch - self.config.start_epoch
        remaining_epochs = self.config.epochs - self.current_epoch
        
        if epochs_completed > 0:
            avg_epoch_time = elapsed_total / epochs_completed
            eta_seconds = avg_epoch_time * remaining_epochs
        else:
            avg_epoch_time = 0
            eta_seconds = 0
            
        return {
            'elapsed_total': elapsed_total,
            'elapsed_formatted': format_time(elapsed_total),
            'avg_epoch_time': avg_epoch_time,
            'eta_seconds': eta_seconds,
            'eta_formatted': format_time(eta_seconds),
            'progress_percent': (self.current_epoch / self.config.epochs) * 100
        }

# ==================== 工具函数 ====================
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def load_checkpoint(model, optimizer, scheduler, filename, load_optimizer_state=True, verbose=True):
    """加载检查点"""
    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if load_optimizer_state and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if load_optimizer_state and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    start_epoch = checkpoint.get('epoch', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    train_loss = checkpoint.get('train_loss', 0)
    val_loss = checkpoint.get('val_loss', 0)
    
    # 获取原始batch size（用于学习率调整）
    original_config = checkpoint.get('config', {})
    original_batch_size = original_config.get('batch_size', None)
    
    if verbose:
        print(f"✅ 检查点已加载: {filename}")
        print(f"   从Epoch {start_epoch}开始继续训练")
        print(f"   当前验证损失: {val_loss:.6f}")
        print(f"   最佳验证损失: {best_val_loss:.6f}")
        if original_batch_size:
            print(f"   原始Batch Size: {original_batch_size}")
    
    return start_epoch, train_loss, val_loss, best_val_loss, original_batch_size

def save_checkpoint(epoch, model, optimizer, scheduler, train_loss, val_loss, best_val_loss, config, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'config': config.__dict__
    }
    torch.save(checkpoint, filename)
    print(f"💾 检查点已保存: {filename}")

# ==================== 训练函数 ====================
def train_pinn_seq2seq(config):
    print("=" * 70)
    print("🚀 PrinterPINN Seq2Seq 训练")
    print("=" * 70)
    
    # 打印配置信息
    print(f"\n📋 训练配置:")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   Gradient Accumulation: {config.gradient_accumulation_steps}")
    print(f"   Effective Batch Size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"   Learning Rate: {config.lr}")
    print(f"   Epochs: {config.epochs}")
    print(f"   Device: {config.device}")
    print(f"   物理损失权重: {config.lambda_physics}")

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # 数据处理
    processor = MemoryDataProcessor(
        config.data_path,
        config.seq_len,
        config.pred_len,
        config.max_samples,
        config
    )

    train_dataset = Seq2SeqDataset(processor.train_X, processor.train_ctrl, processor.train_Y)
    val_dataset = Seq2SeqDataset(processor.val_X, processor.val_ctrl, processor.val_Y)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=seq2seq_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=seq2seq_collate_fn
    )

    # 模型
    model = PrinterPINN_Seq2Seq(config)
    if torch.cuda.device_count() > 1:
        print(f"🎮 使用 {torch.cuda.device_count()} 个 GPU!")
        model = nn.DataParallel(model)
    model = model.to(config.device)

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-5
    )

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=config.warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.epochs - config.warmup_epochs,
        eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.warmup_epochs]
    )

    criterion = nn.MSELoss()
    scaler = GradScaler('cuda')

    # TensorBoard
    log_dir = os.path.join("runs", "seq2seq_experiment")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # 从检查点恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    
    if config.resume_from is not None and os.path.exists(config.resume_from):
        start_epoch, _, _, best_val_loss, original_batch_size = load_checkpoint(
            model, optimizer, scheduler, config.resume_from, 
            config.load_optimizer_state, verbose=True
        )
        
        # 处理batch size变化的情况
        if original_batch_size and original_batch_size != config.batch_size:
            batch_scale = config.batch_size / original_batch_size
            print(f"\n⚠️  检测到Batch Size变化!")
            print(f"   原始: {original_batch_size} -> 当前: {config.batch_size}")
            print(f"   缩放因子: {batch_scale:.2f}x")
            
            # 根据batch size缩放调整学习率和优化器状态
            if config.load_optimizer_state:
                print(f"   🔧 检测到加载了优化器状态，但batch size已变化")
                print(f"   建议设置 --load_optimizer_state=False 或 --lr {config.lr * batch_scale:.2e}")
                print(f"   或手动调整学习率以匹配新的batch size")
            else:
                # 自动调整学习率
                new_lr = config.lr * batch_scale
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"   ✅ 学习率已自动调整: {config.lr:.2e} -> {new_lr:.2e}")
                config.lr = new_lr
            
            # 调整scheduler的milestone
            warmup_scheduler.total_iters = max(1, int(config.warmup_epochs))
            cosine_scheduler.T_max = max(1, config.epochs - config.warmup_epochs)
        
        config.start_epoch = start_epoch
        config.original_batch_size = original_batch_size
    else:
        print("ℹ️  从头开始训练")

    # 创建状态管理器
    state_manager = TrainingStateManager(config, model, optimizer, scheduler, config.checkpoint_dir)
    state_manager.current_epoch = start_epoch
    state_manager.best_val_loss = best_val_loss

    # 注册优雅退出处理器
    def graceful_exit_handler(signum=None, frame=None):
        print(f"\n{'='*70}")
        print(f"⚠️  接收到退出信号，正在优雅关闭...")
        print(f"{'='*70}")
        
        time_info = state_manager.get_time_info()
        print(f"\n📊 训练进度:")
        print(f"   当前Epoch: {state_manager.current_epoch}/{config.epochs}")
        print(f"   训练损失: {state_manager.train_loss:.6f}")
        print(f"   验证损失: {state_manager.val_loss:.6f}")
        print(f"   最佳验证损失: {state_manager.best_val_loss:.6f}")
        print(f"\n⏱️  时间统计:")
        print(f"   已训练时间: {time_info['elapsed_formatted']}")
        print(f"   平均每个Epoch: {format_time(time_info['avg_epoch_time'])}")
        
        if config.save_on_exit:
            print(f"\n💾 正在保存检查点...")
            final_checkpoint_path = os.path.join(
                config.checkpoint_dir, 
                f"interrupted_epoch{state_manager.current_epoch}.pth"
            )
            state_manager.save_checkpoint(final_checkpoint_path)
            
            # 保存训练摘要
            summary_path = os.path.join(config.checkpoint_dir, "training_summary.txt")
            with open(summary_path, 'w') as f:
                f.write("训练中断摘要\n")
                f.write("="*70 + "\n\n")
                f.write(f"当前Epoch: {state_manager.current_epoch}/{config.epochs}\n")
                f.write(f"训练损失: {state_manager.train_loss:.6f}\n")
                f.write(f"验证损失: {state_manager.val_loss:.6f}\n")
                f.write(f"最佳验证损失: {state_manager.best_val_loss:.6f}\n")
                f.write(f"已训练时间: {time_info['elapsed_formatted']}\n")
                f.write(f"配置:\n")
                for key, value in config.__dict__.items():
                    f.write(f"  {key}: {value}\n")
            print(f"   摘要已保存: {summary_path}")
        
        print(f"\n{'='*70}")
        print(f"✅ 优雅关闭完成")
        print(f"{'='*70}\n")
        
        writer.close()
        exit(0)

    # 注册信号处理器
    signal.signal(signal.SIGINT, graceful_exit_handler)
    signal.signal(signal.SIGTERM, graceful_exit_handler)
    
    # 注册atexit处理器（正常退出时也会调用）
    atexit.register(lambda: None)  # 防止重复注册

    # 训练循环
    print_every = 50
    epoch_times = []
    
    print("\n🚀 开始训练...\n")
    print(f"{'='*70}")
    
    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()

        for batch_idx, (batch_x_hist, batch_x_ctrl, batch_y) in enumerate(train_loader):
            batch_x_hist = batch_x_hist.to(config.device)
            batch_x_ctrl = batch_x_ctrl.to(config.device)
            batch_y = batch_y.to(config.device)

            with autocast('cuda'):
                outputs = model(batch_x_hist, batch_x_ctrl)
                
                data_loss = criterion(outputs, batch_y)
                
                if isinstance(model, nn.DataParallel):
                    physics_loss = model.module.physics_loss(outputs, batch_y)
                else:
                    physics_loss = model.physics_loss(outputs, batch_y)
                
                total_loss = data_loss + config.lambda_physics * physics_loss

            scaler.scale(total_loss).backward()
            
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += total_loss.item() * config.gradient_accumulation_steps

            if (batch_idx + 1) % print_every == 0:
                avg_so_far = epoch_loss / (batch_idx + 1)
                time_info = state_manager.get_time_info()
                
                print(f"  🔵 Epoch {epoch+1:3d}/{config.epochs} | "
                      f"Batch {batch_idx+1:5d}/{len(train_loader):5d} | "
                      f"Loss: {avg_so_far:.6f} | LR: {optimizer.param_groups[0]['lr']:.6e} | "
                      f"ETA: {time_info['eta_formatted']}")

        avg_train_loss = epoch_loss / len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x_hist, batch_x_ctrl, batch_y in val_loader:
                batch_x_hist = batch_x_hist.to(config.device)
                batch_x_ctrl = batch_x_ctrl.to(config.device)
                batch_y = batch_y.to(config.device)
                
                with autocast('cuda'):
                    outputs = model(batch_x_hist, batch_x_ctrl)
                    loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        # 更新状态管理器
        state_manager.update_epoch(epoch + 1, avg_train_loss, avg_val_loss)
        time_info = state_manager.get_time_info()
        
        # 打印epoch摘要
        print(f"🟢 Epoch {epoch+1:3d}/{config.epochs} ({time_info['progress_percent']:5.1f}%) | "
              f"Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | "
              f"Time: {epoch_time:.2f}s | ETA: {time_info['eta_formatted']}")

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Time/epoch", epoch_time, epoch)

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_filename = "best_seq2seq_model.pth"
            save_checkpoint(epoch+1, model, optimizer, scheduler, avg_train_loss, avg_val_loss, 
                           best_val_loss, config, os.path.join(config.checkpoint_dir, checkpoint_filename))
            print(f"  💾 最佳模型已保存 (验证损失: {best_val_loss:.6f})")

        # 定期保存
        if (epoch + 1) % config.save_interval == 0:
            checkpoint_filename = f"checkpoint_epoch{epoch+1}.pth"
            save_checkpoint(epoch+1, model, optimizer, scheduler, avg_train_loss, avg_val_loss, 
                           best_val_loss, config, os.path.join(config.checkpoint_dir, checkpoint_filename))

    total_time = time.time() - state_manager.training_start_time
    print(f"\n{'='*70}")
    print(f"🎉 训练完成！")
    print(f"{'='*70}")
    print(f"⏱️  总用时: {format_time(total_time)}")
    print(f"📊 最佳验证损失: {best_val_loss:.6f}")
    print(f"📊 训练损失: {avg_train_loss:.6f}")
    print(f"{'='*70}\n")
    
    writer.close()

def get_args():
    parser = argparse.ArgumentParser(description='训练3D打印机PINN模型')
    parser.add_argument('--data_path', type=str, default='enterprise_dataset/printer_enterprise_data.csv', 
                        help='数据文件路径')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
    parser.add_argument('--resume_from', type=str, help='从指定检查点恢复训练')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_seq2seq', help='检查点保存目录')
    parser.add_argument('--save_on_exit', type=bool, default=True, help='退出时是否保存权重')
    parser.add_argument('--save_interval', type=int, default=5, help='定期保存间隔')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    parser.add_argument('--load_optimizer_state', type=bool, default=True, 
                        help='加载检查点时是否加载优化器状态（batch size变化时建议为False）')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    config = Config()
    
    # 更新配置参数
    config.data_path = args.data_path
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.resume_from = args.resume_from
    config.checkpoint_dir = args.checkpoint_dir
    config.save_on_exit = args.save_on_exit
    config.save_interval = args.save_interval
    config.device = args.device
    config.load_optimizer_state = args.load_optimizer_state
    
    train_pinn_seq2seq(config)
