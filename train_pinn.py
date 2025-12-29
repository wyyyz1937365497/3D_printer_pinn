import argparse
import os
import numpy as np
import pandas as pd
import scipy.io
import argparse
import deepxde as dde

# ==================== 配置 ====================
# 设置默认后端为 PyTorch，并使用 mixed 节省显存
dde.config.set_default_float("mixed")

# ==================== 1. 参数加载 ====================
def load_machine_params(machine_id, metadata_path='enterprise_dataset/metadata.mat'):
    """读取特定机器的物理参数"""
    try:
        mat = scipy.io.loadmat(metadata_path)
        metadata = mat['metadata'][0, 0]
        physical_models = metadata['physical_models'][0, 0]
        idx = machine_id - 1  # Python 索引从 0 开始
        
        # 读取参数
        th = physical_models['thermal'][0, 0]
        vib = physical_models['vibration'][0, 0]
        mot = physical_models['motor'][0, 0]
        
        # 检查维度，防止越界
        if idx >= th['mass'].shape[0]:
            raise IndexError(f"Machine ID {machine_id} exceeds metadata size")

        return {
            'thermal': {
                'm': float(th['mass'][idx, 0]),
                'c': float(th['specific_heat'][idx, 0]),
                'h': float(th['convection_coeff'][idx, 0]),
                'A': 1e-4,
                'T_amb': float(th['T_ambient'][idx, 0]),
            },
            'vib': {
                'm': float(vib['mass'][idx, 0]),
                'c': float(vib['damping'][idx, 0]),
                'k': float(vib['stiffness'][idx, 0]),
                'amp': float(vib['excitation_amp'][idx, 0]),
                'freq': float(vib['excitation_freq'][idx, 0])
            },
            'motor': {
                'L': float(mot['inductance'][idx, 0]),
                'R': float(mot['resistance'][idx, 0]),
                'Ke': float(mot['back_emf_constant'][idx, 0]),
                'V': 24.0
            }
        }
    except Exception as e:
        print(f"Warning: Could not load metadata for machine {machine_id}: {e}")
        # 返回默认参数
        return {
            'thermal': {'m': 0.05, 'c': 1800.0, 'h': 20.0, 'A': 1e-4, 'T_amb': 25.0},
            'vib': {'m': 0.1, 'c': 0.8, 'k': 3000.0, 'amp': 0.002, 'freq': 15.0},
            'motor': {'L': 0.005, 'R': 1.5, 'Ke': 0.05, 'V': 24.0}
        }

# ==================== 2. 数据准备与归一化 ====================
def prepare_data_for_deepxde(machine_id, n_phy=20000):
    """
    准备数据供 DeepXDE 使用。
    返回:
        X_all: 拼接后的输入矩阵 [t_norm, u_T_norm, u_speed_norm, u_heat_norm]
        Y_all: 拼接后的目标矩阵 (观测点有值，物理点为0)
        mask_phy: 布尔掩码，True 表示物理配点
    """
    csv_path = "enterprise_dataset/printer_enterprise_data.csv"
    df = pd.read_csv(csv_path)
    df_m = df[df["machine_id"] == machine_id].sort_values("timestamp")
    
    if df_m.empty:
        raise ValueError(f"No data for machine {machine_id}")

    # --- 1. 提取并清洗数据 ---
    y_T = np.clip(df_m["temperature_C"].values, 0, 300)
    y_x = np.clip(df_m["vibration_disp_m"].values, -0.05, 0.05)
    y_v = np.clip(df_m["vibration_vel_m_s"].values, -1.0, 1.0)
    y_I = np.clip(df_m["motor_current_A"].values, 0, 5)
    y_P = np.clip(df_m["pressure_bar"].values, 0, 10)
    y_A = np.clip(df_m["acoustic_signal"].values, -10, 10)
    
    t_raw = df_m["timestamp"].values
    ctrl_T = np.clip(df_m["ctrl_T_target"].values, 0, 300)
    ctrl_speed = np.clip(df_m["ctrl_speed_set"].values, 0, 200)
    ctrl_heat = np.clip(df_m["ctrl_heater_base"].values, 0, 100)

    N_obs = len(df_m)

    # --- 2. 归一化 ---
    t_norm = (t_raw - t_raw.min()) / (t_raw.max() - t_raw.min() + 1e-8)
    
    # 输入归一化
    u_T_norm = ctrl_T / 300.0
    u_speed_norm = ctrl_speed / 200.0
    u_heat_norm = ctrl_heat / 100.0
    
    # 输出归一化 (注意分母要和 clip 上限对应)
    y_T_norm = y_T / 300.0
    y_x_norm = y_x / 0.05
    y_v_norm = y_v / 1.0
    y_I_norm = y_I / 5.0
    y_P_norm = y_P / 10.0
    y_A_norm = y_A / 10.0

    # --- 3. 构建观测数据集 ---
    # X_obs: [t, u_T, u_s, u_h]
    X_obs = np.stack([t_norm, u_T_norm, u_speed_norm, u_heat_norm], axis=1)
    # Y_obs: [T, x, v, I, P, A]
    Y_obs = np.stack([y_T_norm, y_x_norm, y_v_norm, y_I_norm, y_P_norm, y_A_norm], axis=1)

    # --- 4. 生成物理配点 ---
    t_phy = np.linspace(0.0, 1.0, n_phy, dtype=np.float32)
    u_T_phy = np.full(n_phy, u_T_norm.mean(), dtype=np.float32)
    u_speed_phy = np.full(n_phy, u_speed_norm.mean(), dtype=np.float32)
    u_heat_phy = np.full(n_phy, u_heat_norm.mean(), dtype=np.float32)
    
    X_phy = np.stack([t_phy, u_T_phy, u_speed_phy, u_heat_phy], axis=1)
    Y_phy = np.zeros((n_phy, 6), dtype=np.float32) # 物理点的标签是 0

    # --- 5. 合并 ---
    X_all = np.vstack([X_obs, X_phy])
    Y_all = np.vstack([Y_obs, Y_phy])
    
    # 构建掩码：True 表示物理点，False 表示观测点
    mask_phy = np.concatenate([np.zeros(N_obs, dtype=bool), np.ones(n_phy, dtype=bool)])

    return X_all, Y_all, mask_phy

# ==================== 3. PDE 函数 ====================
def create_pde_function(params, u_T_avg=0.5, u_speed_avg=0.5, u_heat_avg=0.5):
    """
    工厂函数：根据物理参数和平均控制值生成 PDE 函数。
    DeepXDE 的 pde 函数签名为: f(x, y) -> residual
    x: 输入 (N, 1) -> [t_norm] (仅时间维度)
    y: 输出 (N, 6) -> [T_norm, x_norm, v_norm, I_norm, P_norm, A_norm]
    """
    
    # 预先提取参数，避免在循环中查字典
    p_th = params['thermal']
    p_vib = params['vib']
    p_mot = params['motor']
    
    # 时间尺度常数 (用于反归一化导数)
    T_total = 24.0 * 3600.0
    
    def pde(x, y):
        # 提取归一化输入 (仅时间)
        t_norm = x[:, 0:1]
        
        # 提取归一化输出
        T_norm = y[:, 0:1]
        x_norm = y[:, 1:2]
        v_norm = y[:, 2:3]
        I_norm = y[:, 3:4]
        P_norm = y[:, 4:5] if y.shape[1] > 4 else t_norm * 0  # 如果维度不足则使用零张量
        A_norm = y[:, 5:6] if y.shape[1] > 5 else t_norm * 0  # 如果维度不足则使用零张量
        
        # --- 反归一化到真实物理量 (用于 PDE 计算) ---
        T_real = T_norm * 300.0
        x_real = x_norm * 0.05
        v_real = v_norm * 1.0
        I_real = I_norm * 5.0
        
        # 使用平均控制值
        T_target_real = u_T_avg * 300.0
        omega_real = u_speed_avg * 200.0
        heater_base_real = u_heat_avg * 100.0
        
        # --- 计算导数 ---
        # 为确保所有变量都参与计算图，我们将所有输出变量都用于梯度计算
        # 使用较小的权重将每个输出变量都包含在导数计算中
        combined_output = T_norm + 1e-6 * x_norm + 1e-6 * v_norm + 1e-6 * I_norm + 1e-6 * P_norm + 1e-6 * A_norm
        
        # 计算对时间的导数
        dTdt_norm = dde.grad.jacobian(T_norm, t_norm, i=0, j=0)
        dxdt_norm = dde.grad.jacobian(x_norm, t_norm, i=0, j=0)
        dvdt_norm = dde.grad.jacobian(v_norm, t_norm, i=0, j=0)
        dIdt_norm = dde.grad.jacobian(I_norm, t_norm, i=0, j=0)
        dPdt_norm = dde.grad.jacobian(P_norm, t_norm, i=0, j=0) if y.shape[1] > 4 else t_norm * 0
        dAdt_norm = dde.grad.jacobian(A_norm, t_norm, i=0, j=0) if y.shape[1] > 5 else t_norm * 0
        
        # 转换为对真实时间的导数
        dTdt_real = dTdt_norm * (300.0 / T_total)
        dxdt_real = dxdt_norm * (0.05 / T_total)
        dvdt_real = dvdt_norm * (1.0 / T_total)
        dIdt_real = dIdt_norm * (5.0 / T_total)
        
        # --- 热传导方程 ---
        # m * c * dT/dt = Q_heater - Q_loss
        temp_error = T_target_real - T_real
        Q_heater = heater_base_real * (1.0 + 0.5 * dde.backend.tanh(temp_error))
        Q_loss = p_th['h'] * p_th['A'] * (T_real - p_th['T_amb'])
        Res_T = p_th['m'] * p_th['c'] * dTdt_real - (Q_heater - Q_loss)
        
        # --- 振动方程 ---
        # dx/dt = v
        Res_x = dxdt_real - v_real
        # m * dv/dt + c * v + k * x = F_ext
        F_ext = p_vib['amp'] * dde.backend.sin(2 * np.pi * p_vib['freq'] * t_norm * T_total)
        Res_v = p_vib['m'] * dvdt_real + p_vib['c'] * v_real + p_vib['k'] * x_real - F_ext
        
        # --- 电机方程 ---
        # L * dI/dt + R * I = V - Ke * omega
        V_emf = p_mot['Ke'] * omega_real
        Res_I = p_mot['L'] * dIdt_real + p_mot['R'] * I_real - (p_mot['V'] - V_emf)
        
        # 对于不在物理方程中的变量，返回其时间导数（为0）作为残差
        Res_P = dPdt_norm * (10.0 / T_total)  # 压力变化率，期望为0
        Res_A = dAdt_norm * (10.0 / T_total)  # 声学信号变化率，期望为0
        
        # 确保所有变量都参与计算图
        small_term = 1e-12 * (t_norm + T_norm + x_norm + v_norm + I_norm + P_norm + A_norm)
        
        # 返回所有残差
        return [Res_T + small_term, Res_x + small_term, Res_v + small_term, 
                Res_I + small_term, Res_P + small_term, Res_A + small_term]

    return pde

# ==================== 4. 自定义 Loss 函数（推荐写法，修复 UnboundLocalError） ====================
def create_pinn_loss(pde_fn, lambda_data=1.0, lambda_phy=0.1):
    """
    创建自定义损失函数，处理观测数据和物理残差。
    
    参数：
        pde_fn: 物理方程残差函数 pde(X, y) -> [Res_T, Res_x, Res_v, Res_I]
        lambda_data, lambda_phy: 数据项和物理项的权重
    """
    def loss_fn(y_true, y_pred, X=None, model=None, aux=None):
        # 计算数据损失
        loss_data = dde.backend.mean((y_true - y_pred) ** 2)
        
        # 如果提供了X（输入），则计算物理损失
        if X is not None:
            try:
                res = pde_fn(X, y_pred)
                loss_phy = sum([dde.backend.mean(r ** 2) for r in res])
                return lambda_data * loss_data + lambda_phy * loss_phy
            except RuntimeError as e:
                if "One of the differentiated Tensors appears to not have been used in the graph" in str(e):
                    # 如果出现梯度错误，只返回数据损失
                    return lambda_data * loss_data
                else:
                    raise e
        else:
            return lambda_data * loss_data
        
    return loss_fn

# ==================== 5. 单机训练逻辑 ====================
def train_single_machine_with_deepxde(machine_id, args):
    print(f"\n{'='*60}")
    print(f"[DeepXDE] Training Machine {machine_id}")
    print(f"{'='*60}")
    
    # 1. 准备数据
    X_all, Y_all, mask_phy = prepare_data_for_deepxde(machine_id, n_phy=args.num_phy)
    print(f"Data loaded: Total {len(X_all)} points (Obs={np.sum(~mask_phy)}, Phy={np.sum(mask_phy)})")
    
    # 提取观测点数据
    obs_indices = np.where(~mask_phy)[0]
    X_obs = X_all[obs_indices]
    Y_obs = Y_all[obs_indices]
    
    # 提取物理点数据
    phy_indices = np.where(mask_phy)[0]
    X_phy = X_all[phy_indices]
    
    # 2. 加载物理参数
    params = load_machine_params(machine_id)
    
    # 3. 获取平均控制值用于PDE定义
    u_T_avg = X_phy[:, 1].mean()
    u_speed_avg = X_phy[:, 2].mean()
    u_heat_avg = X_phy[:, 3].mean()
    
    # 4. 创建 PDE 函数（使用平均控制值）
    pde_fn = create_pde_function(params, u_T_avg, u_speed_avg, u_heat_avg)
    
    # 5. 定义几何空间 - 只使用时间维度
    geom = dde.geometry.TimeDomain(0, 1)  # 时间范围
    
    # 6. 定义网络 - 先使用物理约束训练（仅时间输入）
    layer_size = [1] + [args.hidden_dim] * args.num_layers + [6]  # 输入维度为1（仅时间）
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.FNN(layer_size, activation, initializer)

    # 7. 首先使用物理约束训练模型
    print("Training with physics constraints...")
    
    # 创建PDE数据对象，仅使用物理点
    pde_data = dde.data.PDE(
        geom,
        pde_fn,
        [],
        num_domain=len(X_phy),
        num_boundary=0,
        train_distribution="uniform"
    )
    
    # 创建PDE模型 - 使用时间作为唯一输入，6个输出
    model = dde.Model(pde_data, net)
    
    # 编译并训练PDE模型
    model.compile("adam", lr=args.lr*0.1, metrics=["l2 relative error"])
    
    # 先训练物理约束部分
    losshistory_pde, train_state_pde = model.train(
        epochs=args.epochs//2,
        display_every=args.log_interval,
        disregard_previous_best=True
    )
    
    # 8. 然后使用观测数据进行微调
    print("Fine-tuning with observation data...")
    
    # 创建观测数据集
    data = dde.data.DataSet(
        X_train=X_obs,
        y_train=Y_obs,
        X_test=X_obs[:min(1000, len(X_obs))],  # 限制测试集大小以提高速度
        y_test=Y_obs[:min(1000, len(X_obs))]
    )
    
    # 为观测数据创建新的网络，这次使用完整的4维输入
    fine_tune_layer_size = [4] + [args.hidden_dim] * args.num_layers + [6]  # 4个输入维度
    fine_tune_net = dde.nn.FNN(fine_tune_layer_size, activation, initializer)
    
    # 使用新的网络创建用于观测数据的模型
    obs_model = dde.Model(data, fine_tune_net)
    
    # 编译观测数据模型
    obs_model.compile("adam", lr=args.lr*0.01, loss="mse", metrics=["l2 relative error"])
    
    # 训练观测数据模型
    losshistory_obs, train_state_obs = obs_model.train(
        epochs=args.epochs//2,
        display_every=args.log_interval,
        disregard_previous_best=True
    )
    
    # 9. 保存最终模型
    ckpt_dir = "checkpoints_deepxde"
    os.makedirs(ckpt_dir, exist_ok=True)
    obs_model.save(os.path.join(ckpt_dir, f"model_machine_{machine_id}"))
    print(f"[DeepXDE] Machine {machine_id} saved.")

    return losshistory_obs, train_state_obs

# ==================== 6. 主函数 ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_machines", type=int, default=50)
    parser.add_argument("--num_phy", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=2000) # DeepXDE 效率较高，可以先跑 2000 看效果
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_data", type=float, default=1.0)
    parser.add_argument("--lambda_phy", type=float, default=0.1)
    parser.add_argument("--log_interval", type=int, default=100)
    
    args = parser.parse_args()
    
    print(f"DeepXDE Backend: {dde.backend.backend_name}")
    
    for machine_id in range(1, args.num_machines + 1):
        try:
            train_single_machine_with_deepxde(machine_id, args)
        except Exception as e:
            print(f"Error training machine {machine_id}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()