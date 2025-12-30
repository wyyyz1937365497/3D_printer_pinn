%% =========================================================================
%  文件名: data_prepare.m (完整版 - 支持故障数据生成)
%  功能: 
%    1. 生成正常运行数据
%    2. 生成故障数据用于测试
%    3. 使用稳定的积分器算法
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

fprintf('=== 生成完整数据集 (正常+故障) ===\n');
fprintf('时长: %d小时 | 步长: %.3fs | 机器数: %d\n', ...
    config.simulation_hours, config.dt, config.n_machines);

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

% 预计算电机隐式求解系数
motor_inv_dt = 1 / config.dt;
motor_tau_inv = motor_model.resistance' ./ motor_model.inductance';
motor_denom = motor_inv_dt + motor_tau_inv;

fprintf('正在进行正常仿真...\n');
for t = 2:N_steps
    current_time = time_vector(t);
    
    % --- 热传导 ---
    prev_temp = temperature(t-1, :);
    temp_error = thermal_model.T_target' - prev_temp;
    heater_power = thermal_model.heater_power_base' .* (1 + 0.5*tanh(temp_error));
    heat_loss = thermal_model.convection_coeff' .* (prev_temp - thermal_model.T_ambient');
    dTdt = (heater_power - heat_loss) ./ (thermal_model.mass' .* thermal_model.specific_heat');
    temperature(t, :) = prev_temp + dTdt * config.dt;
    
    % --- 振动 ---
    prev_disp = vibration_disp(t-1, :);
    prev_vel = vibration_vel(t-1, :);
    excitation = vibration_model.excitation_amp' .* sin(2*pi*vibration_model.excitation_freq'*current_time);
    
    accel = (excitation - vibration_model.damping'.*prev_vel - vibration_model.stiffness'.*prev_disp) ./ vibration_model.mass';
    new_vel = prev_vel + accel * config.dt;
    new_disp = prev_disp + new_vel * config.dt;
    
    vibration_vel(t, :) = new_vel;
    vibration_disp(t, :) = new_disp;
    
    % --- 电机电流 ---
    prev_current = motor_current(t-1, :);
    speed_base = 100; 
    back_emf = motor_model.back_emf_constant' * speed_base;
    voltage_source = 24; 
    
    motor_term1 = prev_current .* motor_inv_dt;
    motor_term2 = (voltage_source - back_emf) ./ motor_model.inductance';
    motor_current(t, :) = (motor_term1 + motor_term2) ./ motor_denom;
    
    % --- 挤出压力 ---
    extrusion_pressure(t, :) = extrusion_model.base_pressure' + 0.1*randn(1, config.n_machines);
    
    % --- 声学 ---
    acoustic_signal(t, :) = 0.1*randn(1, config.n_machines) + sin(2*pi*acoustic_model.base_freq'*current_time);
end

%% ==================== 4. 生成故障数据 ==========================
fprintf('正在生成故障数据...\n');

% 选择部分机器添加故障
faulty_machines = randperm(config.n_machines, 10); % 10%的机器有故障

for mid = faulty_machines
    fault_start = randi([N_steps/2, N_steps*0.8]); % 随机故障开始时间
    fault_type = randi([1, 3]); % 随机故障类型
    
    switch fault_type
        case 1 % 加热器故障
            fprintf('机器 %d: 加热器故障 (从步 %d 开始)\n', mid, fault_start);
            temperature(fault_start:end, mid) = temperature(fault_start:end, mid) * 0.7; % 加热效率降低
            
        case 2 % 阻尼增加 (轴承磨损)
            fprintf('机器 %d: 轴承磨损故障 (从步 %d 开始)\n', mid, fault_start);
            original_damping = vibration_model.damping(mid);
            for t = fault_start:N_steps
                excitation = vibration_model.excitation_amp(mid) * sin(2*pi*vibration_model.excitation_freq(mid)*time_vector(t));
                prev_disp = vibration_disp(t-1, mid);
                prev_vel = vibration_vel(t-1, mid);
                
                % 增加阻尼
                fault_damping = original_damping * 1.5;
                accel = (excitation - fault_damping*prev_vel - vibration_model.stiffness(mid)*prev_disp) / vibration_model.mass(mid);
                new_vel = prev_vel + accel * config.dt;
                new_disp = prev_disp + new_vel * config.dt;
                
                vibration_vel(t, mid) = new_vel;
                vibration_disp(t, mid) = new_disp;
            end
            
        case 3 % 电机故障 (电流异常)
            fprintf('机器 %d: 电机故障 (从步 %d 开始)\n', mid, fault_start);
            motor_current(fault_start:end, mid) = motor_current(fault_start:end, mid) * 1.3; % 电流增大
    end
end

%% ==================== 5. 导出数据 ==========================
% 创建故障标签列
fault_label = zeros(N_steps, config.n_machines);
for mid = faulty_machines
    % 找到该机器的故障开始时间
    fault_start_idx = find(~(temperature(:, mid) == temperature(end, mid)), 1, 'last');
    if isempty(fault_start_idx)
        fault_start_idx = N_steps/2;
    end
    fault_label(fault_start_idx:end, mid) = 1;
end

% 组合数据
ctrl_T_target = repmat(thermal_model.T_target', N_steps, 1);
ctrl_speed_set = 100 * ones(N_steps, config.n_machines); 
ctrl_heater_base = repmat(thermal_model.heater_power_base', N_steps, 1);

[time_grid, machine_grid] = ndgrid(time_vector, 1:config.n_machines);

data_matrix = [time_grid(:), machine_grid(:), ...
               ctrl_T_target(:), ctrl_speed_set(:), ctrl_heater_base(:), ...
               temperature(:), vibration_disp(:), vibration_vel(:), ...
               motor_current(:), extrusion_pressure(:), acoustic_signal(:), ...
               fault_label(:)];

column_names = {'timestamp', 'machine_id', ...
                'ctrl_T_target', 'ctrl_speed_set', 'ctrl_heater_base', ...
                'temperature_C', 'vibration_disp_m', 'vibration_vel_m_s', ...
                'motor_current_A', 'pressure_bar', 'acoustic_signal', ...
                'fault_label'};

T = array2table(data_matrix, 'VariableNames', column_names);

csv_path = fullfile(config.output_dir, 'printer_enterprise_data.csv');
writetable(T, csv_path);

% 保存元数据
metadata = struct();
metadata.physical_models.thermal = thermal_model;
metadata.physical_models.vibration = vibration_model;
metadata.physical_models.motor = motor_model;
metadata.faulty_machines = faulty_machines;
save(fullfile(config.output_dir, 'metadata.mat'), 'metadata');

fprintf('数据已保存: %s\n', csv_path);
fprintf('总样本数: %d (正常+故障)\n', size(data_matrix, 1));
fprintf('Done!\n');
