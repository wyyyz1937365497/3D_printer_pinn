%% =========================================================================
%  文件名: data_prepare.m (终极修复版)
%  修复策略:
%    1. 振动: 使用半隐式欧拉法 解决数值发散
%    2. 电机: 使用隐式欧拉法 解决刚性方程发散
%    3. 保持 dt=0.01 (控制文件大小，同时利用稳定算法)
% =========================================================================

clear; clc; close all;

%% ==================== 1. 配置 ==========================
config = struct();
config.simulation_hours = 1;   % 仿真时长 1小时
config.dt = 0.01;              % 时间步长 10ms
config.T_total = config.simulation_hours * 3600;
config.n_machines = 50;
config.output_dir = 'enterprise_dataset';

if ~exist(config.output_dir, 'dir')
    mkdir(config.output_dir);
end

fprintf('=== 生成高稳定性数据集 (积分器升级版) ===\n');
fprintf('时长: %d小时 | 步长: %.3fs | 机器数: %d\n', ...
    config.simulation_hours, config.dt, config.n_machines);

%% ==================== 2. 物理参数生成 ==========================
rng(2024);

% 热参数 (显式欧拉即可，变化慢)
thermal_model = struct();
thermal_model.T_ambient = 22 + 3*rand(config.n_machines, 1);
thermal_model.T_target = 210 + 5*rand(config.n_machines, 1);
thermal_model.mass = 0.05 + 0.01*rand(config.n_machines, 1);
thermal_model.specific_heat = 1800 + 200*rand(config.n_machines, 1);
thermal_model.convection_coeff = 15 + 5*rand(config.n_machines, 1);
thermal_model.heater_power_base = 40 + 5*rand(config.n_machines, 1);

% 振动参数 (使用半隐式欧拉)
vibration_model = struct();
vibration_model.mass = 0.1 + 0.02*rand(config.n_machines, 1);
vibration_model.stiffness = 3000 + 500*rand(config.n_machines, 1);
vibration_model.damping = 0.8 + 0.3*rand(config.n_machines, 1);
vibration_model.excitation_freq = 15 + 5*rand(config.n_machines, 1);
vibration_model.excitation_amp = 0.002 + 0.001*rand(config.n_machines, 1);

% 电机参数 (使用隐式欧拉)
motor_model = struct();
motor_model.rated_current = 2.0 + 0.2*rand(config.n_machines, 1);
motor_model.resistance = 1.5 + 0.2*rand(config.n_machines, 1);
motor_model.inductance = 0.005 + 0.001*rand(config.n_machines, 1);
motor_model.back_emf_constant = 0.05 + 0.01*rand(config.n_machines, 1);

% 挤出参数
extrusion_model = struct();
extrusion_model.base_pressure = 5 + 1*rand(config.n_machines, 1);
extrusion_model.viscosity = 1000 + 200*rand(config.n_machines, 1);

% 声学参数
acoustic_model = struct();
acoustic_model.base_freq = 2000 + 200*rand(config.n_machines, 1);

%% ==================== 3. 仿真循环 (积分器升级) ==========================
N_steps = ceil(config.T_total / config.dt);
time_vector = (0:config.dt:(N_steps-1)*config.dt)';

% 预分配
temperature = zeros(N_steps, config.n_machines);
vibration_disp = zeros(N_steps, config.n_machines);
vibration_vel = zeros(N_steps, config.n_machines);
motor_current = zeros(N_steps, config.n_machines);
extrusion_pressure = zeros(N_steps, config.n_machines);
acoustic_signal = zeros(N_steps, config.n_machines);

% 初始条件
temperature(1, :) = thermal_model.T_ambient';

fprintf('正在进行稳定仿真...\n');

% 预计算电机隐式求解系数 (提升速度)
% 隐式公式: I_new = (I_old/dt + V/L) / (1/dt + R/L)
motor_inv_dt = 1 / config.dt;
motor_tau_inv = motor_model.resistance' ./ motor_model.inductance'; % R/L
motor_denom = motor_inv_dt + motor_tau_inv; % 1/dt + R/L

for t = 2:N_steps
    current_time = time_vector(t);
    
    % --- 热传导 (显式，慢过程) ---
    prev_temp = temperature(t-1, :);
    temp_error = thermal_model.T_target' - prev_temp;
    heater_power = thermal_model.heater_power_base' .* (1 + 0.5*tanh(temp_error));
    heat_loss = thermal_model.convection_coeff' .* (prev_temp - thermal_model.T_ambient');
    dTdt = (heater_power - heat_loss) ./ (thermal_model.mass' .* thermal_model.specific_heat');
    temperature(t, :) = prev_temp + dTdt * config.dt;
    
    % --- 振动 (半隐式欧拉: 先算速度，用新速度算位置) ---
    prev_disp = vibration_disp(t-1, :);
    prev_vel = vibration_vel(t-1, :);
    excitation = vibration_model.excitation_amp' .* sin(2*pi*vibration_model.excitation_freq'*current_time);
    
    % 1. 计算加速度
    accel = (excitation - vibration_model.damping'.*prev_vel - vibration_model.stiffness'.*prev_disp) ./ vibration_model.mass';
    % 2. 更新速度
    new_vel = prev_vel + accel * config.dt;
    % 3. 使用新速度更新位置 (关键步骤！)
    new_disp = prev_disp + new_vel * config.dt;
    
    vibration_vel(t, :) = new_vel;
    vibration_disp(t, :) = new_disp;
    
    % --- 电机电流 (隐式欧拉: 无条件稳定) ---
    prev_current = motor_current(t-1, :);
    speed_base = 100; 
    back_emf = motor_model.back_emf_constant' * speed_base;
    voltage_source = 24; 
    
    % 隐式更新公式: I_new = (I_old/dt + (V-BackEMF)/L) / (1/dt + R/L)
    motor_term1 = prev_current .* motor_inv_dt;
    motor_term2 = (voltage_source - back_emf) ./ motor_model.inductance';
    motor_current(t, :) = (motor_term1 + motor_term2) ./ motor_denom;
    
    % --- 挤出压力 ---
    extrusion_pressure(t, :) = extrusion_model.base_pressure' + 0.1*randn(1, config.n_machines);
    
    % --- 声学 ---
    acoustic_signal(t, :) = 0.1*randn(1, config.n_machines) + sin(2*pi*acoustic_model.base_freq'*current_time);
end
fprintf('仿真完成，检查数据稳定性...\n');

%% ==================== 4. 稳定性检查与导出 ==========================
% 快速检查是否有异常值
if any(isnan([temperature(:); vibration_disp(:); motor_current(:)]))
    error('仿真失败：检测到 NaN，请检查算法');
end
if any(isinf([temperature(:); vibration_disp(:); motor_current(:)]))
    error('仿真失败：检测到 Inf，请检查算法');
end
fprintf('数据检查通过，无 NaN/Inf。\n');

fprintf('正在导出 CSV...\n');
ctrl_T_target = repmat(thermal_model.T_target', N_steps, 1);
ctrl_speed_set = 100 * ones(N_steps, config.n_machines); 
ctrl_heater_base = repmat(thermal_model.heater_power_base', N_steps, 1);

[time_grid, machine_grid] = ndgrid(time_vector, 1:config.n_machines);

% 为了避免内存溢出，分块创建 Table (对于 1800万行数据，直接创建 Table 会很慢)
% 这里我们直接写入文本流，或者使用优化的 writetable
column_names = {'timestamp', 'machine_id', ...
                'ctrl_T_target', 'ctrl_speed_set', 'ctrl_heater_base', ...
                'temperature_C', 'vibration_disp_m', 'vibration_vel_m_s', ...
                'motor_current_A', 'pressure_bar', 'acoustic_signal'};

% 组合数据矩阵
data_matrix = [time_grid(:), machine_grid(:), ...
               ctrl_T_target(:), ctrl_speed_set(:), ctrl_heater_base(:), ...
               temperature(:), vibration_disp(:), vibration_vel(:), ...
               motor_current(:), extrusion_pressure(:), acoustic_signal(:)];

% 创建 Table
T = array2table(data_matrix, 'VariableNames', column_names);

csv_path = fullfile(config.output_dir, 'printer_enterprise_data.csv');
writetable(T, csv_path);

fprintf('CSV 已保存: %s\n', csv_path);

%% ==================== 5. 元数据 ==========================
metadata = struct();
metadata.physical_models = struct();
metadata.physical_models.thermal = thermal_model;
metadata.physical_models.vibration = vibration_model;
metadata.physical_models.motor = motor_model;
save(fullfile(config.output_dir, 'metadata.mat'), 'metadata');

fprintf('Done!\n');
