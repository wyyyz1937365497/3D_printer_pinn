%% =========================================================================
%  文件名: data_prepare.m
%  功能: 企业级3D打印机预测性维护数据生成系统 (适配Python训练)
%  特性: 50台机器 x 6传感器 x 24小时 = 完整数据集
%  日期: 2024
% =========================================================================

clear; clc; close all;

%% ==================== 1. 配置 ==========================
config = struct();
config.simulation_hours = 24;
config.dt = 0.5;
config.T_total = config.simulation_hours * 3600;
config.n_machines = 50;
config.output_dir = 'enterprise_dataset';

if ~exist(config.output_dir, 'dir')
    mkdir(config.output_dir);
end

fprintf('Generating data for %d machines...\n', config.n_machines);

%% ==================== 2. 物理参数生成 ==========================
rng(2024);

% 热参数
thermal_model = struct();
thermal_model.T_ambient = 22 + 3*rand(config.n_machines, 1);
thermal_model.T_target = 210 + 5*rand(config.n_machines, 1);
thermal_model.mass = 0.05 + 0.01*rand(config.n_machines, 1);
thermal_model.specific_heat = 1800 + 200*rand(config.n_machines, 1);
thermal_model.convection_coeff = 15 + 5*rand(config.n_machines, 1);
thermal_model.heater_power_base = 40 + 5*rand(config.n_machines, 1);

% 振动参数
vibration_model = struct();
vibration_model.mass = 0.1 + 0.02*rand(config.n_machines, 1);
vibration_model.stiffness = 3000 + 500*rand(config.n_machines, 1);
vibration_model.damping = 0.8 + 0.3*rand(config.n_machines, 1);
vibration_model.excitation_freq = 15 + 5*rand(config.n_machines, 1);
vibration_model.excitation_amp = 0.002 + 0.001*rand(config.n_machines, 1);

% 电机参数
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

%% ==================== 3. 仿真循环 ==========================
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

fprintf('Simulating physics...\n');
for t = 2:N_steps
    current_time = time_vector(t);
    
    % 热传导
    prev_temp = temperature(t-1, :);
    temp_error = thermal_model.T_target' - prev_temp;
    heater_power = thermal_model.heater_power_base' .* (1 + 0.5*tanh(temp_error));
    heat_loss = thermal_model.convection_coeff' .* (prev_temp - thermal_model.T_ambient');
    dTdt = (heater_power - heat_loss) ./ (thermal_model.mass' .* thermal_model.specific_heat');
    temperature(t, :) = prev_temp + dTdt * config.dt;
    
    % 振动 (简化欧拉法)
    prev_disp = vibration_disp(t-1, :);
    prev_vel = vibration_vel(t-1, :);
    excitation = vibration_model.excitation_amp' .* sin(2*pi*vibration_model.excitation_freq'*current_time);
    accel = (excitation - vibration_model.damping'.*prev_vel - vibration_model.stiffness'.*prev_disp) ./ vibration_model.mass';
    vibration_vel(t, :) = prev_vel + accel * config.dt;
    vibration_disp(t, :) = prev_disp + vibration_vel(t, :) * config.dt;
    
    % 电机电流
    prev_current = motor_current(t-1, :);
    speed_base = 100; % 假设基础速度
    back_emf = motor_model.back_emf_constant' * speed_base;
    dIdt = (24 - back_emf - motor_model.resistance'.*prev_current) ./ motor_model.inductance';
    motor_current(t, :) = prev_current + dIdt * config.dt;
    
    % 挤出压力
    extrusion_pressure(t, :) = extrusion_model.base_pressure' + 0.1*randn(1, config.n_machines);
    
    % 声学
    acoustic_signal(t, :) = 0.1*randn(1, config.n_machines) + sin(2*pi*acoustic_model.base_freq'*current_time);
end

%% ==================== 4. 数据导出 (CSV) ==========================
fprintf('Exporting CSV...\n');

% 生成控制变量
% ctrl_T_target: 每台机器的目标温度
ctrl_T_target = repmat(thermal_model.T_target', N_steps, 1);
% ctrl_speed_set: 假设恒定为 100 (可在此处改为随时间变化)
ctrl_speed_set = 100 * ones(N_steps, config.n_machines); 
% ctrl_heater_base: 每台机器的基础加热功率
ctrl_heater_base = repmat(thermal_model.heater_power_base', N_steps, 1);

% 构建完整数据矩阵
[time_grid, machine_grid] = ndgrid(time_vector, 1:config.n_machines);

% 写入表头
column_names = {'timestamp', 'machine_id', ...
                'ctrl_T_target', 'ctrl_speed_set', 'ctrl_heater_base', ...
                'temperature_C', 'vibration_disp_m', 'vibration_vel_m_s', ...
                'motor_current_A', 'pressure_bar', 'acoustic_signal'};

% 创建表并写入 CSV
T = table('Size', [length(time_grid(:)), length(column_names)], 'VariableTypes', repmat({'double'}, 1, length(column_names)), 'VariableNames', column_names);

% 赋值给表
T.timestamp = time_grid(:);
T.machine_id = machine_grid(:);
T.ctrl_T_target = ctrl_T_target(:);
T.ctrl_speed_set = ctrl_speed_set(:);
T.ctrl_heater_base = ctrl_heater_base(:);
T.temperature_C = temperature(:);
T.vibration_disp_m = vibration_disp(:);
T.vibration_vel_m_s = vibration_vel(:);
T.motor_current_A = motor_current(:);
T.pressure_bar = extrusion_pressure(:);
T.acoustic_signal = acoustic_signal(:);

writetable(T, fullfile(config.output_dir, 'printer_enterprise_data.csv'));

%% ==================== 5. 元数据保存 (.mat) 【新增部分】 ==========================
fprintf('Saving Metadata (metadata.mat)...\n');

% 构建元数据结构，与 Python 读取逻辑匹配
metadata = struct();
metadata.physical_models = struct();

% 将生成的参数模型赋值给 metadata
% 注意：这里直接赋值即可，Python 会通过索引读取
metadata.physical_models.thermal = thermal_model;
metadata.physical_models.vibration = vibration_model;
metadata.physical_models.motor = motor_model;

% 保存到 mat 文件
save(fullfile(config.output_dir, 'metadata.mat'), 'metadata');

fprintf('Done! All files saved to %s\n', config.output_dir);
