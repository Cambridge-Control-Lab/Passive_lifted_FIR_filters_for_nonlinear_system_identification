smatalb = settings; % dimensions: scalar settings object.
smatalb.matlab.appearance.figure.GraphicsTheme.TemporaryValue = 'light'; % dimensions: scalar string setting.
clear variables;
close all;
clc;

script_dir = fileparts(mfilename('fullpath')); % dimensions: scalar string, script folder or temporary run folder.
pwd_dir = pwd; % dimensions: scalar string, MATLAB current folder.
candidate_exp1_dirs = { ...
    fileparts(script_dir), ...
    fullfile(pwd_dir, 'Exp1'), ...
    pwd_dir, ...
    fileparts(pwd_dir), ...
    fileparts(fileparts(pwd_dir))}; % dimensions: 1 x 5 cell array of scalar string paths.
exp1_dir = ''; % dimensions: scalar string, resolved Exp1 folder.
for i_candidate = 1:numel(candidate_exp1_dirs)
    candidate_exp1_dir = candidate_exp1_dirs{i_candidate}; % dimensions: scalar string path.
    if exist(fullfile(candidate_exp1_dir, 'Training_data'), 'dir') && exist(fullfile(candidate_exp1_dir, 'utility_functions'), 'dir')
        exp1_dir = candidate_exp1_dir;
        break;
    end
end
if isempty(exp1_dir)
    error('Could not locate Exp1 folder. Set MATLAB current folder to the repository root or Exp1 before running Exp1_N4SID.m.');
end
data_dir = fullfile(exp1_dir, 'Training_data'); % dimensions: scalar string, training data folder.
results_dir = fullfile(exp1_dir, 'Results'); % dimensions: scalar string, output results folder.
utility_dir = fullfile(exp1_dir, 'utility_functions'); % dimensions: scalar string, MATLAB utility folder.
addpath(utility_dir);

load(fullfile(data_dir, 'Data_M_NLdamper_500B_OneCart.mat'));
dta_clear = dta;
load(fullfile(data_dir, 'Data_M_NLdamper_500B_OneCart_SNR10.mat'));
dta_noise = dta;
Ts = dta_clear.Ts;
n_train = 70;
u_sysid = squeeze(dta_clear.ipt_train_mat(1,:,1:n_train));
y_sysid = squeeze(dta_clear.opt_train_mat(1,:,1:n_train));
u_sysid_noise = squeeze(dta_noise.ipt_train_mat(1,:,1:n_train));
y_sysid_noise = squeeze(dta_noise.opt_train_mat(1,:,1:n_train));
num_exp = size(u_sysid, 2);
%% 
data_cell = cell(num_exp, 1);
data_cell_noise = cell(num_exp, 1);
for k = 1:num_exp
    y_k = y_sysid(:, k);
    u_k = u_sysid(:, k);

    y_k_noise = y_sysid_noise(:, k);
    u_k_noise = u_sysid_noise(:, k);
    data_cell{k} = iddata(y_k, u_k, Ts);
    data_cell_noise{k} = iddata(y_k_noise, u_k_noise, Ts);
end
data_sysid_clear = merge(data_cell{:});
data_sysid_noise = merge(data_cell_noise{:});

%% 


opt = n4sidOptions;
opt.Focus = 'simulation';

opt.EnforceStability = true;


%% Save N4SID results for best-box comparison

n_order_n4sid = 20; % dimensions: scalar, final N4SID model order.
n_sample_n4sid = size(dta_clear.ipt_train_mat, 2); % dimensions: scalar, number of time samples per batch.
n_batch_n4sid = size(dta_clear.ipt_train_mat, 3); % dimensions: scalar, number of batches.
split_train_idx = 1:n_train; % dimensions: 1 x 300, train comparison batch indices.
split_test_idx = 401:500; % dimensions: 1 x 100, test comparison batch indices.

u_clear_all = reshape(dta_clear.ipt_train_mat(1,:,:), n_sample_n4sid, n_batch_n4sid); % dimensions: 250 x 500.
y_clear_all = reshape(dta_clear.opt_train_mat(1,:,:), n_sample_n4sid, n_batch_n4sid); % dimensions: 250 x 500.
u_noise_all = reshape(dta_noise.ipt_train_mat(1,:,:), n_sample_n4sid, n_batch_n4sid); % dimensions: 250 x 500.
y_noise_all = reshape(dta_noise.opt_train_mat(1,:,:), n_sample_n4sid, n_batch_n4sid); % dimensions: 250 x 500.

disp('Training final clean N4SID model');
ss_clear_final = n4sid(data_sysid_clear, n_order_n4sid, opt, 'DisturbanceModel', 'none'); % dimensions: scalar idss model.
disp('Training final noisy N4SID model');
ss_noise_final = n4sid(data_sysid_noise, n_order_n4sid, opt, 'DisturbanceModel', 'none'); % dimensions: scalar idss model.
%% 
mean(ss_clear_final.Report.Fit.FitPercent)
mean(ss_noise_final.Report.Fit.FitPercent)
%% 

u_clear_train_batch = u_clear_all(:, split_train_idx); % dimensions: 250 x 300.
y_clear_train_batch = y_clear_all(:, split_train_idx); % dimensions: 250 x 300.
u_clear_test_batch = u_clear_all(:, split_test_idx); % dimensions: 250 x 100.
y_clear_test_batch = y_clear_all(:, split_test_idx); % dimensions: 250 x 100.

u_noise_train_batch = u_noise_all(:, split_train_idx); % dimensions: 250 x 300.
y_noise_train_batch = y_noise_all(:, split_train_idx); % dimensions: 250 x 300.
u_noise_test_batch = u_noise_all(:, split_test_idx); % dimensions: 250 x 100.
y_noise_test_batch = y_noise_all(:, split_test_idx); % dimensions: 250 x 100.

disp('Comparing clean N4SID batches');
data_clear_train_compare = iddata(num2cell(y_clear_train_batch, 1), num2cell(u_clear_train_batch, 1), Ts); % dimensions: multi-experiment iddata, 300 batches.
data_clear_test_compare = iddata(num2cell(y_clear_test_batch, 1), num2cell(u_clear_test_batch, 1), Ts); % dimensions: multi-experiment iddata, 100 batches.
y_clear_pred_train_cell = compare(data_clear_train_compare, ss_clear_final); % dimensions: 1 x 300 cell, predicted iddata.
y_clear_pred_test_cell = compare(data_clear_test_compare, ss_clear_final); % dimensions: 1 x 100 cell, predicted iddata.
y_clear_pred_train = cell2mat(cellfun(@(z) z.OutputData, y_clear_pred_train_cell, 'UniformOutput', false)); % dimensions: 250 x 300.
y_clear_pred_test = cell2mat(cellfun(@(z) z.OutputData, y_clear_pred_test_cell, 'UniformOutput', false)); % dimensions: 250 x 100.
%% 

disp('Comparing noisy N4SID batches');
data_noise_train_compare = iddata(num2cell(y_noise_train_batch, 1), num2cell(u_noise_train_batch, 1), Ts); % dimensions: multi-experiment iddata, 300 batches.
data_noise_test_compare = iddata(num2cell(y_noise_test_batch, 1), num2cell(u_noise_test_batch, 1), Ts); % dimensions: multi-experiment iddata, 100 batches.
y_noise_pred_train_cell = compare(data_noise_train_compare, ss_noise_final); % dimensions: 1 x 300 cell, predicted iddata.
y_noise_pred_test_cell = compare(data_noise_test_compare, ss_noise_final); % dimensions: 1 x 100 cell, predicted iddata.
y_noise_pred_train = cell2mat(cellfun(@(z) z.OutputData, y_noise_pred_train_cell, 'UniformOutput', false)); % dimensions: 250 x 300.
y_noise_pred_test = cell2mat(cellfun(@(z) z.OutputData, y_noise_pred_test_cell, 'UniformOutput', false)); % dimensions: 250 x 100.

%% Compute N4SID fitting metrics

[fit_clear_train, mse_clear_train] = func_new_metrics_by_traj(y_clear_train_batch, y_clear_pred_train); % dimensions: 300 x 1.
[fit_clear_test, mse_clear_test] = func_new_metrics_by_traj(y_clear_test_batch, y_clear_pred_test); % dimensions: 100 x 1.
[fit_noise_train, mse_noise_train] = func_new_metrics_by_traj(y_noise_train_batch, y_noise_pred_train); % dimensions: 300 x 1.
[fit_noise_test, mse_noise_test] = func_new_metrics_by_traj(y_noise_test_batch, y_noise_pred_test); % dimensions: 100 x 1.

fprintf('N4SID clean train mean fit = %.4f, test mean fit = %.4f\n', ...
    mean(fit_clear_train), mean(fit_clear_test));
fprintf('N4SID noisy train mean fit = %.4f, test mean fit = %.4f\n', ...
    mean(fit_noise_train), mean(fit_noise_test));

run_Exp1_n4sid_train = struct; % dimensions: scalar struct.
run_Exp1_n4sid_train.schema_version = 'n4sid_v1'; % dimensions: scalar string.
run_Exp1_n4sid_train.mode = 'n4sid'; % dimensions: scalar string.
run_Exp1_n4sid_train.run_name = 'run_Exp1_n4sid_train'; % dimensions: scalar string.
run_Exp1_n4sid_train.Ts = Ts; % dimensions: scalar.
run_Exp1_n4sid_train.n_order = n_order_n4sid; % dimensions: scalar.
run_Exp1_n4sid_train.n_train = n_train; % dimensions: scalar.
run_Exp1_n4sid_train.split_train_idx = split_train_idx; % dimensions: 1 x 300.
run_Exp1_n4sid_train.split_test_idx = split_test_idx; % dimensions: 1 x 100.
run_Exp1_n4sid_train.ss_model = ss_clear_final; % dimensions: scalar idss model.
run_Exp1_n4sid_train.y_train_batch = y_clear_train_batch; % dimensions: 250 x 300.
run_Exp1_n4sid_train.y_pre_train_batch = y_clear_pred_train; % dimensions: 250 x 300.
run_Exp1_n4sid_train.y_pre_train_batch_cl = y_clear_pred_train; % dimensions: 250 x 300.
run_Exp1_n4sid_train.y_test_batch = y_clear_test_batch; % dimensions: 250 x 100.
run_Exp1_n4sid_train.y_pre_test_batch = y_clear_pred_test; % dimensions: 250 x 100.
run_Exp1_n4sid_train.y_pre_test_batch_cl = y_clear_pred_test; % dimensions: 250 x 100.
run_Exp1_n4sid_train.fit_train = fit_clear_train; % dimensions: 300 x 1.
run_Exp1_n4sid_train.mse_train = mse_clear_train; % dimensions: 300 x 1.
run_Exp1_n4sid_train.fit_test = fit_clear_test; % dimensions: 100 x 1.
run_Exp1_n4sid_train.mse_test = mse_clear_test; % dimensions: 100 x 1.
run_Exp1_n4sid_train.fit_train_mean = mean(fit_clear_train); % dimensions: scalar.
run_Exp1_n4sid_train.fit_test_mean = mean(fit_clear_test); % dimensions: scalar.
run_Exp1_n4sid_train.mse_train_mean = mean(mse_clear_train); % dimensions: scalar.
run_Exp1_n4sid_train.mse_test_mean = mean(mse_clear_test); % dimensions: scalar.

run_Exp1_n4sid_SNR10_train = struct; % dimensions: scalar struct.
run_Exp1_n4sid_SNR10_train.schema_version = 'n4sid_v1'; % dimensions: scalar string.
run_Exp1_n4sid_SNR10_train.mode = 'n4sid_SNR10'; % dimensions: scalar string.
run_Exp1_n4sid_SNR10_train.run_name = 'run_Exp1_n4sid_SNR10_train'; % dimensions: scalar string.
run_Exp1_n4sid_SNR10_train.Ts = Ts; % dimensions: scalar.
run_Exp1_n4sid_SNR10_train.n_order = n_order_n4sid; % dimensions: scalar.
run_Exp1_n4sid_SNR10_train.n_train = n_train; % dimensions: scalar.
run_Exp1_n4sid_SNR10_train.split_train_idx = split_train_idx; % dimensions: 1 x 300.
run_Exp1_n4sid_SNR10_train.split_test_idx = split_test_idx; % dimensions: 1 x 100.
run_Exp1_n4sid_SNR10_train.ss_model = ss_noise_final; % dimensions: scalar idss model.
run_Exp1_n4sid_SNR10_train.y_train_batch = y_noise_train_batch; % dimensions: 250 x 300.
run_Exp1_n4sid_SNR10_train.y_pre_train_batch = y_noise_pred_train; % dimensions: 250 x 300.
run_Exp1_n4sid_SNR10_train.y_pre_train_batch_cl = y_noise_pred_train; % dimensions: 250 x 300.
run_Exp1_n4sid_SNR10_train.y_test_batch = y_noise_test_batch; % dimensions: 250 x 100.
run_Exp1_n4sid_SNR10_train.y_pre_test_batch = y_noise_pred_test; % dimensions: 250 x 100.
run_Exp1_n4sid_SNR10_train.y_pre_test_batch_cl = y_noise_pred_test; % dimensions: 250 x 100.
run_Exp1_n4sid_SNR10_train.fit_train = fit_noise_train; % dimensions: 300 x 1.
run_Exp1_n4sid_SNR10_train.mse_train = mse_noise_train; % dimensions: 300 x 1.
run_Exp1_n4sid_SNR10_train.fit_test = fit_noise_test; % dimensions: 100 x 1.
run_Exp1_n4sid_SNR10_train.mse_test = mse_noise_test; % dimensions: 100 x 1.
run_Exp1_n4sid_SNR10_train.fit_train_mean = mean(fit_noise_train); % dimensions: scalar.
run_Exp1_n4sid_SNR10_train.fit_test_mean = mean(fit_noise_test); % dimensions: scalar.
run_Exp1_n4sid_SNR10_train.mse_train_mean = mean(mse_noise_train); % dimensions: scalar.
run_Exp1_n4sid_SNR10_train.mse_test_mean = mean(mse_noise_test); % dimensions: scalar.

if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end
save(fullfile(results_dir, 'run_Exp1_n4sid_train.mat'), 'run_Exp1_n4sid_train');
save(fullfile(results_dir, 'run_Exp1_n4sid_SNR10_train.mat'), 'run_Exp1_n4sid_SNR10_train');
