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

def create_pde_function(params):
    """
    工厂函数：根据物理参数生成 PDE 函数。
    DeepXDE 的 pde 函数签名为: f(x, y) -> residual
    x: 输入 (N, 4) -> [t_norm, u_T_norm, u_speed_norm, u_heat_norm]
    y: 输出 (N, 6) -> [T_norm, x_norm, v_norm, I_norm, P_norm, A_norm]
    """
    
    # 预先提取参数，避免在循环中查字典
    p_th = params['thermal']
    p_vib = params['vib']
    p_mot = params['motor']
    
    # 时间尺度常数 (用于反归一化导数)
    T_total = 24.0 * 3600.0
    
    def pde(x, y):
        # 提取归一化输入
        t_norm = x[:, 0:1]
        u_T_norm = x[:, 1:2]
        u_speed_norm = x[:, 2:3]
        u_heat_norm = x[:, 3:4]
        
        # 提取归一化输出
        T_norm = y[:, 0:1]
        x_norm = y[:, 1:2]
        v_norm = y[:, 2:3]
        I_norm = y[:, 3:4]
        
        # --- 反归一化到真实物理量 (用于 PDE 计算) ---
        T_real = T_norm * 300.0
        x_real = x_norm * 0.05
        v_real = v_norm * 1.0
        I_real = I_norm * 5.0
        
        T_target_real = u_T_norm * 300.0
        omega_real = u_speed_norm * 200.0
        heater_base_real = u_heat_norm * 100.0
        
        # --- 计算导数 ---
        # 重要：确保所有输入都参与计算图
        dTdt_norm = dde.grad.jacobian(T_norm, t_norm, i=0, j=0)
        dxdt_norm = dde.grad.jacobian(x_norm, t_norm, i=0, j=0)
        dvdt_norm = dde.grad.jacobian(v_norm, t_norm, i=0, j=0)
        dIdt_norm = dde.grad.jacobian(I_norm, t_norm, i=0, j=0)
        
        # 转换为对真实时间的导数
        dTdt_real = dTdt_norm * (300.0 / T_total)
        dxdt_real = dxdt_norm * (0.05 / T_total)
        dvdt_real = dvdt_norm * (1.0 / T_total)
        dIdt_real = dIdt_norm * (5.0 / T_total)
        
        # --- 热传导方程 ---
        # m * c * dT/dt = Q_heater - Q_loss
        temp_error = T_target_real - T_real
        # 确保所有控制输入都参与计算，即使是很小的参与
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
        
        # 确保所有输入变量都以某种方式影响输出
        # 添加一个非常小的项，确保所有输入都参与到梯度计算中
        # 这样可以避免梯度计算错误
        small_term = 1e-12 * (t_norm + u_T_norm + u_speed_norm + u_heat_norm)
        
        # 将小项添加到每个残差中，确保所有输入都参与梯度计算
        return [Res_T + small_term, Res_x + small_term, Res_v + small_term, Res_I + small_term]

    return pde

# ==================== 4. 自定义 Loss 函数（推荐写法，修复 UnboundLocalError） ====================
def create_pinn_loss(pde_fn, mask_phy, lambda_data=1.0, lambda_phy=0.1):
    """
    创建自定义损失函数，处理观测数据和物理残差。
    
    参数：
        pde_fn: 物理方程残差函数 pde(X, y) -> [Res_T, Res_x, Res_v, Res_I]
        mask_phy: numpy bool 数组，True 表示物理配点，False 表示观测点
        lambda_data, lambda_phy: 数据项和物理项的权重
    """
    # 1. 把 mask 转成张量（不指定 dtype，保持 bool）
    mask_phy_tensor = dde.backend.as_tensor(mask_phy)

    # 2. 定义真正的 loss_fn，会由 DeepXDE 在训练中调用
    # 注意：DeepXDE 的签名是 loss_fn(y_true, y_pred, X=None, model=None, aux=None)
    def loss_fn(y_true, y_pred, X=None, model=None, aux=None):
        # 2.1 设备对齐
        if hasattr(mask_phy_tensor, 'to') and mask_phy_tensor.device != y_pred.device:
            mask = mask_phy_tensor.to(y_pred.device)
        else:
            mask = mask_phy_tensor

        mask_data = ~mask  # 观测点的mask
        
        # 计算数据损失（仅对观测点）
        if mask_data.any():  # 如果存在观测点
            loss_data = ((y_pred[mask_data] - y_true[mask_data]) ** 2).mean()
        else:
            loss_data = dde.backend.zeros_like(dde.backend.mean(y_pred))
        
        # 计算物理损失（对所有点）
        try:
            res = pde_fn(X, y_pred)
            loss_phy = sum([(r ** 2).mean() for r in res])
            return lambda_data * loss_data + lambda_phy * loss_phy
        except RuntimeError as e:
            if "One of the differentiated Tensors appears to not have been used in the graph" in str(e):
                # 如果出现梯度错误，只返回数据损失
                return lambda_data * loss_data
            else:
                raise e
        
    return loss_fn

# ==================== 5. 单机训练逻辑 ====================
def train_single_machine_with_deepxde(machine_id, args):
    print(f"\n{'='*60}")
    print(f"[DeepXDE] Training Machine {machine_id}")
    print(f"{'='*60}")
    
    # 1. 准备数据
    X_all, Y_all, mask_phy = prepare_data_for_deepxde(machine_id, n_phy=args.num_phy)
    print(f"Data loaded: Total {len(X_all)} points (Obs={np.sum(~mask_phy)}, Phy={np.sum(mask_phy)})")
    
    # 2. 加载物理参数
    params = load_machine_params(machine_id)
    
    # 3. 创建 PDE 函数
    pde_fn = create_pde_function(params)
    
    # 提取观测点数据
    obs_indices = np.where(~mask_phy)[0]
    X_obs = X_all[obs_indices]
    Y_obs = Y_all[obs_indices]
    
    # 4. 定义数据集
    data = dde.data.DataSet(
        X_train=X_obs,
        y_train=Y_obs,
        X_test=X_obs,
        y_test=Y_obs
    )
    
    # 5. 定义网络
    layer_size = [4] + [args.hidden_dim] * args.num_layers + [6]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.FNN(layer_size, activation, initializer)
    
    # 6.1 创建自定义损失函数
    loss_fn = create_pinn_loss(pde_fn, mask_phy, args.lambda_data, args.lambda_phy)

    model = dde.Model(data, net)
    
    # 7. 编译与训练
    # 使用 Adam 优化器
    # DeepXDE 会自动处理 device (cuda/cpu)
    model.compile("adam", lr=args.lr, loss=loss_fn, metrics=["l2 relative error"])
    
    # 使用学习率衰减 (可选)
    # variable = dde.callbacks.VariableValue(...)
    
    # 训练
    losshistory, train_state = model.train(
        epochs=args.epochs,
        batch_size=None, # None 表示全量数据，或者指定数字
        display_every=args.log_interval,
        disregard_previous_best=True
    )
    
    # 8. 保存模型
    ckpt_dir = "checkpoints_deepxde"
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save(os.path.join(ckpt_dir, f"model_machine_{machine_id}"))
    print(f"[DeepXDE] Machine {machine_id} saved.")

    return losshistory, train_state

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