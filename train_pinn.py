import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
import scipy.io
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.autograd import grad

# ==================== 1. 物理方程定义 ====================
def physics_residual_strict(t, u, y, params):
    if not t.requires_grad:
        t = t.requires_grad_(True)

    # 这里的 y 已经是反归一化后的真实物理量了
    T = y[:, 0:1]
    x = y[:, 1:2]
    v = y[:, 2:3]
    I = y[:, 3:4]
    
    # 注意：这里的 u 还是归一化后的，需要在物理方程里也反归一化回来
    # 或者，更简单的方法：在 Dataset 里只归一化了 u，
    # 而物理方程里用到 T_target (u[:, 0]) 和 heater_base (u[:, 2])。
    # 所以我们要把 u 的这两列乘回去。
    
    T_target = u[:, 0:1] * 300.0   # 反归一化
    # u[:, 1] 是 speed，在振动方程里没用到原始值，只用到了 omega = u[:, 1] (如果之前代码是这么写的)
    # 但之前的代码 omega = 100.0 * u[:, 1:2] 是假设 u 没归一化。
    # 现在的 u 已经归一化了，所以 omega 应该是:
    omega_base = 200.0 # 对应上面的 / 200.0
    
    # 修正后的 omega 计算：
    omega = omega_base * u[:, 1:2] 
    
    heater_base_set = u[:, 2:3] * 100.0 # 反归一化

    # --- 热传导方程 ---
    p_th = params['thermal']
    m_th = p_th['m']
    c_th = p_th['c']
    h_th = p_th['h']
    A_th = p_th['A']
    T_amb = p_th['T_amb']
    
    temp_error = T_target - T
    Q_heater = heater_base_set * (1.0 + 0.5 * torch.tanh(temp_error))
    Q_loss = h_th * A_th * (T - T_amb)
    
    dTdt = grad(T, t, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    Res_T = dTdt - (Q_heater - Q_loss) / (m_th * c_th)

    # --- 振动方程 ---
    p_vib = params['vib']
    m_vib = p_vib['m']
    c_vib = p_vib['c']
    k_vib = p_vib['k']
    vib_amp = p_vib['amp']
    vib_freq = p_vib['freq']
    
    dxdt = grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    Res_x = dxdt - v
    
    F_ext = vib_amp * torch.sin(2 * np.pi * vib_freq * t)
    dvdt = grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    Res_v = m_vib * dvdt + c_vib * v + k_vib * x - F_ext

    # --- 电机方程 ---
    p_mot = params['motor']
    L_mot = p_mot['L']
    R_mot = p_mot['R']
    Ke_mot = p_mot['Ke']
    V_sup = p_mot['V']
    
    V_emf = Ke_mot * omega
    dIdt = grad(I, t, grad_outputs=torch.ones_like(I), create_graph=True)[0]
    Res_I = L_mot * dIdt + R_mot * I - (V_sup - V_emf)

    return [Res_T, Res_x, Res_v, Res_I]

# ==================== 2. 网络模型定义 ====================
class PrinterProcessPINN(nn.Module):
    def __init__(self, hidden_layers=[128, 128, 128]):
        super().__init__()
        layers = []
        in_dim = 4
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, 6))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ==================== 3. 数据加载与参数读取 ====================
def load_machine_params(machine_id, metadata_path='enterprise_dataset/metadata.mat'):
    try:
        mat = scipy.io.loadmat(metadata_path)
        metadata = mat['metadata'][0, 0]
        physical_models = metadata['physical_models'][0, 0]
        
        # 修正索引逻辑：MATLAB保存的是 (50, 1)，Python读取通常也是
        # 这里的 idx 应该是 machine_id - 1
        idx = machine_id - 1
        
        th = physical_models['thermal'][0, 0]
        vib = physical_models['vibration'][0, 0]
        mot = physical_models['motor'][0, 0]
        
        # 检查维度是否越界
        if idx >= th['mass'].shape[0]:
            raise IndexError(f"Machine ID {machine_id} exceeds metadata size {th['mass'].shape[0]}")

        params_thermal = {
            'm': float(th['mass'][idx, 0]),
            'c': float(th['specific_heat'][idx, 0]),
            'h': float(th['convection_coeff'][idx, 0]),
            'A': 1e-4,  # 常数
            'T_amb': float(th['T_ambient'][idx, 0]),
            'P_base': float(th['heater_power_base'][idx, 0])
        }

        params_vib = {
            'm': float(vib['mass'][idx, 0]),
            'c': float(vib['damping'][idx, 0]),
            'k': float(vib['stiffness'][idx, 0]),
            'amp': float(vib['excitation_amp'][idx, 0]),
            'freq': float(vib['excitation_freq'][idx, 0])
        }

        params_motor = {
            'L': float(mot['inductance'][idx, 0]),
            'R': float(mot['resistance'][idx, 0]),
            'Ke': float(mot['back_emf_constant'][idx, 0]),
            'V': 24.0
        }

        return {
            'thermal': params_thermal,
            'vib': params_vib,
            'motor': params_motor
        }
    except Exception as e:
        print(f"Warning: Error loading metadata for machine {machine_id}: {e}")
        # 返回默认参数
        return {
            'thermal': {'m': 0.05, 'c': 1800.0, 'h': 20.0, 'A': 1e-4, 'T_amb': 25.0, 'P_base': 40.0},
            'vib': {'m': 0.1, 'c': 0.8, 'k': 3000.0, 'amp': 0.002, 'freq': 15.0},
            'motor': {'L': 0.005, 'R': 1.5, 'Ke': 0.05, 'V': 24.0}
        }

class PrinterDataset(Dataset):
    def __init__(self, machine_id, csv_path="enterprise_dataset/printer_enterprise_data.csv"):
        df = pd.read_csv(csv_path)
        df_m = df[df["machine_id"] == machine_id].sort_values("timestamp")
        
        if df_m.empty:
            raise ValueError(f"No data found for machine_id {machine_id}")

        self.N = len(df_m)
        
        self.t_raw = df_m["timestamp"].values
        self.ctrl_T = df_m["ctrl_T_target"].values
        self.ctrl_speed = df_m["ctrl_speed_set"].values
        self.ctrl_heat = df_m["ctrl_heater_base"].values
        
        self.y_T = df_m["temperature_C"].values
        self.y_x = df_m["vibration_disp_m"].values
        self.y_v = df_m["vibration_vel_m_s"].values
        self.y_I = df_m["motor_current_A"].values
        self.y_P = df_m["pressure_bar"].values
        self.y_A = df_m["acoustic_signal"].values
        
        self.t_min, self.t_max = self.t_raw.min(), self.t_raw.max()
        self.t_norm = (self.t_raw - self.t_min) / (self.t_max - self.t_min + 1e-8)
        
        self.N_phy = 20000
        self.t_phy = np.linspace(0.0, 1.0, self.N_phy, dtype=np.float32)
        self.u_phy_T = np.full(self.N_phy, self.ctrl_T.mean(), dtype=np.float32)
        self.u_phy_speed = np.full(self.N_phy, self.ctrl_speed.mean(), dtype=np.float32)
        self.u_phy_heat = np.full(self.N_phy, self.ctrl_heat.mean(), dtype=np.float32)

    def __len__(self):
        return self.N + self.N_phy

    def __getitem__(self, idx):
        if idx < self.N:
            t = torch.tensor([self.t_norm[idx]], dtype=torch.float32)
            
            # --- 关键修改：对输入 u 进行粗糙归一化 ---
            # 目标温度约210，速度约100，加热功率约40
            # 除以各自的最大估值，使其大致在 [0,1] 区间
            u_T_norm = self.ctrl_T[idx] / 300.0
            u_speed_norm = self.ctrl_speed[idx] / 200.0
            u_heat_norm = self.ctrl_heat[idx] / 100.0
            u = torch.tensor([u_T_norm, u_speed_norm, u_heat_norm], dtype=torch.float32)
            
            # --- 关键修改：对标签 y 进行粗糙归一化 ---
            # 温度~200，振动~1e-3，电流~2，压力~5，声学~?
            y_T_norm = self.y_T[idx] / 300.0
            y_x_norm = self.y_x[idx] / 0.01   # 假设最大振动 1cm
            y_v_norm = self.y_v[idx] / 0.1    # 假设最大速度 0.1m/s
            y_I_norm = self.y_I[idx] / 5.0
            y_P_norm = self.y_P[idx] / 10.0
            y_A_norm = self.y_A[idx] / 10.0
            
            y = torch.tensor([y_T_norm, y_x_norm, y_v_norm, y_I_norm, y_P_norm, y_A_norm], dtype=torch.float32)
            
            is_obs = 1.0
        else:
            idx2 = idx - self.N
            t = torch.tensor([self.t_phy[idx2]], dtype=torch.float32)
            
            # 物理配点也要用归一化后的控制量均值
            u_T_norm = self.u_phy_T[idx2] / 300.0
            u_speed_norm = self.u_phy_speed[idx2] / 200.0
            u_heat_norm = self.u_phy_heat[idx2] / 100.0
            u = torch.tensor([u_T_norm, u_speed_norm, u_heat_norm], dtype=torch.float32)
            
            # 物理配点没有标签，给 0 (网络会输出归一化后的 0)
            y = torch.zeros(6, dtype=torch.float32)
            is_obs = 0.0
        return t, u, y, is_obs

def collate_fn(batch):
    t_list, u_list, y_list, is_obs_list = zip(*batch)
    return (torch.stack(t_list), 
            torch.stack(u_list), 
            torch.stack(y_list), 
            torch.tensor(is_obs_list).unsqueeze(1))

# ==================== 4. 训练逻辑 ====================
def train_one_epoch_strict(model, optimizer, dataloader, device, params, 
                           epoch, # 新增：当前 epoch 数
                           warmup_epochs=100, # 新增：预热轮数
                           lambda_data=1.0, 
                           lambda_physics=1.0):
    model.train()
    total_loss = 0.0
    total_data_loss = 0.0
    total_phy_loss = 0.0
    
    # 如果在预热期，强制物理损失权重为 0
    current_lambda_physics = 0.0 if epoch < warmup_epochs else lambda_physics
    
    optimizer_stepped = False # 标记 optimizer 是否被调用过，修复 scheduler 警告

    for t_batch, u_batch, y_batch, is_obs_batch in dataloader:
        t_batch = t_batch.to(device)
        if not t_batch.requires_grad:
            t_batch.requires_grad_(True)
            
        u_batch = u_batch.to(device)
        y_batch = y_batch.to(device)
        is_obs_batch = is_obs_batch.to(device)
        
        optimizer.zero_grad()
        
        # 1. 前向传播
        x_input = torch.cat([t_batch, u_batch], dim=1)
        y_pred_norm = model(x_input) # 网络输出的是归一化后的值
        
        # 2. 数据损失 (比较归一化值)
        mask_obs = (is_obs_batch > 0.5).squeeze(-1)
        if mask_obs.any():
            loss_data = nn.MSELoss()(y_pred_norm[mask_obs], y_batch[mask_obs])
        else:
            loss_data = torch.tensor(0.0, device=device)
        
        # 3. 物理损失 (需要反归一化)
        if current_lambda_physics > 0:
            # 反归一化：把预测值还原回真实物理量级
            # 顺序: T, x, v, I, P, A
            y_pred_T = y_pred_norm[:, 0:1] * 300.0
            y_pred_x = y_pred_norm[:, 1:2] * 0.01
            y_pred_v = y_pred_norm[:, 2:3] * 0.1
            y_pred_I = y_pred_norm[:, 3:4] * 5.0
            # P 和 A 暂时不用物理方程，但也拼回去保持形状
            y_pred_P = y_pred_norm[:, 4:5] * 10.0
            y_pred_A = y_pred_norm[:, 5:6] * 10.0
            
            y_pred_real = torch.cat([y_pred_T, y_pred_x, y_pred_v, y_pred_I, y_pred_P, y_pred_A], dim=1)
            
            Res_T, Res_x, Res_v, Res_I = physics_residual_strict(t_batch, u_batch, y_pred_real, params)

            loss_phy_T = (Res_T ** 2).mean()
            loss_phy_x = (Res_x ** 2).mean()
            loss_phy_v = (Res_v ** 2).mean()
            loss_phy_I = (Res_I ** 2).mean()

            loss_physics = loss_phy_T + loss_phy_x + loss_phy_v + loss_phy_I
        else:
            loss_physics = torch.tensor(0.0, device=device)

        # 4. 总损失
        loss = lambda_data * loss_data + current_lambda_physics * loss_physics

        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: NaN/Inf loss detected in this batch, skipping backward.")
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer_stepped = True
        
        total_loss += loss.item()
        total_data_loss += loss_data.item()
        total_phy_loss += loss_physics.item()
        
    n_batches = len(dataloader)
    if n_batches == 0:
        return 0.0, 0.0, 0.0, optimizer_stepped
    return (total_loss / n_batches, 
            total_data_loss / n_batches, 
            total_phy_loss / n_batches,
            optimizer_stepped)

def find_max_batch_size(model, dataset, device, initial_batch_size=4096, max_batch_size=100000):
    """动态找到最大的 batch_size 以占满显存"""
    batch_size = initial_batch_size
    
    while batch_size <= max_batch_size:
        try:
            # 创建一个小的临时 loader 测试
            test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
            
            # 只测试第一个 batch
            for t_batch, u_batch, y_batch, is_obs_batch in test_loader:
                t_batch = t_batch.to(device)
                if not t_batch.requires_grad:
                    t_batch.requires_grad_(True)
                u_batch = u_batch.to(device)
                y_batch = y_batch.to(device)
                
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                optimizer.zero_grad()
                
                x_input = torch.cat([t_batch, u_batch], dim=1)
                y_pred = model(x_input)
                
                mask_obs = (is_obs_batch > 0.5).squeeze(-1)
                if mask_obs.any():
                    loss = nn.MSELoss()(y_pred[mask_obs], y_batch[mask_obs])
                else:
                    loss = nn.MSELoss()(y_pred, y_batch)
                
                loss.backward()
                break
            
            # 检查显存
            mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
            mem_total = torch.cuda.get_device_properties(device).total_memory / 1024**3
            
            print(f"    Batch {batch_size}: Mem {mem_reserved:.2f}GB / {mem_total:.2f}GB")
            
            if mem_reserved > mem_total * 0.9:
                break
            
            batch_size = int(batch_size * 1.5)
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                batch_size = int(batch_size / 2)
                if batch_size < 256:
                    return 256
            else:
                raise e
    
    return int(batch_size / 1.5)

def train_multiple_machines_on_gpu(gpu_id, machine_list, args):
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    
    model_states = {}
    optimal_batch_size = 4096
    
    for i, machine_id in enumerate(machine_list):
        print(f"\n[GPU {gpu_id}] ======== Machine {machine_id} ({i+1}/{len(machine_list)}) ========")
        
        params = load_machine_params(machine_id)
        model = PrinterProcessPINN(hidden_layers=args.hidden_layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
        
        dataset = PrinterDataset(machine_id, csv_path=args.csv_path)
        
        # 只在第一台机器上寻找最优 batch size
        if i == 0:
            print(f"[GPU {gpu_id}] Finding optimal batch size...")
            optimal_batch_size = find_max_batch_size(model, dataset, device, initial_batch_size=args.batch_size)
            print(f"[GPU {gpu_id}] Using batch_size={optimal_batch_size}")
        
        dataloader = DataLoader(dataset, 
                                batch_size=optimal_batch_size, 
                                shuffle=True, 
                                collate_fn=collate_fn, 
                                num_workers=args.num_workers,
                                pin_memory=True)
        
        # ... (模型、优化器定义不变) ...
        
        best_loss = float('inf')
        for epoch in range(args.epochs):
            # 传入 epoch 参数
            loss, d_loss, p_loss, optimizer_stepped = train_one_epoch_strict(
                model, optimizer, dataloader, device, params,
                epoch=epoch, # 新增
                warmup_epochs=100, # 前100轮只拟合数据
                lambda_data=args.lambda_data,
                lambda_physics=args.lambda_physics
            )
            
            # 只有当 optimizer.step() 被调用过，才调用 scheduler.step()
            if optimizer_stepped:
                scheduler.step()
            
            if epoch % args.log_interval == 0:
                phy_status = "Warmup (Data only)" if epoch < 100 else "Physics ON"
                print(f"[GPU {gpu_id}] M{machine_id} Ep{epoch} [{phy_status}]: L={loss:.4f} (D={d_loss:.4f}, P={p_loss:.4f})")
                
            if epoch % args.save_interval == 0 and epoch > 0:
                if loss < best_loss:
                    best_loss = loss
                    ckpt_path = f"checkpoints/pinn_machine_{machine_id}.pt"
                    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                    torch.save(model.state_dict(), ckpt_path)
        
        # 保存最终状态用于平均
        model_states[machine_id] = model.state_dict()
        print(f"[GPU {gpu_id}] M{machine_id} Finished. Best Loss: {best_loss:.4f}")
    
    return model_states

def average_model_weights(model_states):
    keys = list(model_states.values())[0].keys()
    avg_state_dict = {}
    for key in keys:
        params = [state[key] for state in model_states.values()]
        avg_param = torch.stack(params, dim=0).mean(dim=0)
        avg_state_dict[key] = avg_param
    return avg_state_dict

# ==================== 5. 主函数 ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_machines", type=int, default=50)
    parser.add_argument("--csv_path", type=str, default="enterprise_dataset/printer_enterprise_data.csv")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--hidden_layers", type=int, nargs='+', default=[128, 128, 128])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_data", type=float, default=1.0)
    parser.add_argument("--lambda_physics", type=float, default=0.1)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("Error: No CUDA devices found.")
        return

    print(f"\nSystem Info: {num_gpus} GPUs detected.")
    print(f"Training Config: {args.num_machines} Machines, {args.epochs} Epochs each.\n")

    # 分配任务
    machine_ids = list(range(1, args.num_machines + 1))
    gpu_assignments = {}
    
    # 简单分配：例如 2张卡，每张25台
    machines_per_gpu = args.num_machines // num_gpus
    for i in range(num_gpus):
        start = i * machines_per_gpu
        end = (i + 1) * machines_per_gpu if i < num_gpus - 1 else args.num_machines
        gpu_assignments[i] = machine_ids[start:end]
        print(f"GPU {i}: Assigned {len(gpu_assignments[i])} machines ({start+1}-{end})")

    # 设置多进程
    if os.name != 'nt':
        mp.set_start_method('spawn', force=True)
    
    processes = []
    for gpu_id, machine_list in gpu_assignments.items():
        p = mp.Process(target=train_multiple_machines_on_gpu, args=(gpu_id, machine_list, args))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    print("\n" + "="*60)
    print("All Training Jobs Finished!")
    print("="*60)
    
    # 整合模型
    print("\nMerging models...")
    all_states = []
    for mid in range(1, args.num_machines + 1):
        path = f"checkpoints/pinn_machine_{mid}.pt"
        if os.path.exists(path):
            all_states.append(torch.load(path))
        else:
            print(f"Warning: {path} not found.")
    
    if all_states:
        avg_state = average_model_weights({i: s for i, s in enumerate(all_states)})
        final_model = PrinterProcessPINN(hidden_layers=args.hidden_layers)
        final_model.load_state_dict(avg_state)
        torch.save(final_model.state_dict(), "checkpoints/pinn_unified_model.pt")
        print("Unified model saved to checkpoints/pinn_unified_model.pt")

if __name__ == "__main__":
    main()
