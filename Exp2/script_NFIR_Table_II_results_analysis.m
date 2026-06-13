smatalb = settings;
smatalb.matlab.appearance.figure.GraphicsTheme.TemporaryValue= 'light'; % Can be set to auto (default), light, or dark

clearvars -except selected_case_override;
clc;
close all;

script_path = mfilename('fullpath');
if isempty(script_path)
    script_dir = pwd;
else
    script_dir = fileparts(script_path);
end
candidate_exp2_dirs = { ...
    script_dir, ...
    fileparts(script_dir), ...
    pwd, ...
    fileparts(pwd), ...
    fullfile(pwd, 'Exp2') ...
};
exp2_dir = '';
for i_dir = 1:numel(candidate_exp2_dirs)
    candidate_dir = candidate_exp2_dirs{i_dir};
    if exist(fullfile(candidate_dir, 'Results'), 'dir') && ...
            exist(fullfile(candidate_dir, 'Training_data'), 'dir')
        exp2_dir = candidate_dir;
        break;
    end
end
if isempty(exp2_dir)
    error('Could not locate the Exp2 directory. Run this script from the repository root or the Exp2 directory.');
end
results_dir = fullfile(exp2_dir, 'Results');
utility_dir = fullfile(exp2_dir, 'utility_functions');
addpath(utility_dir);

load(fullfile(results_dir, 'Data_Exp2_Greybox.mat'));
load(fullfile(results_dir, 'run_Exp2_FIR_final_m500_zc50_train.mat'));

load(fullfile(results_dir, 'run_Exp2_150_final_h8_ad01_m500_b3_nonoise_pas_ite4_nobptt_l220_l2NN5_firParallOn_ep20_zc50_neoit_s13_train.mat'));
load(fullfile(results_dir, 'run_Exp2_150_final_h8_ad01_m500_b3_nonoise_pas_ite4_nobptt_l220_l2NN5_firParallOn_ep20_zc50_neoit_s23_train.mat'));


run_tag_list = {'run_Exp2_150_final_h8_ad01_m500_b3_nonoise_pas_ite4_nobptt_l220_l2NN5_firParallOn_ep20_zc50_neoit_s13_train'};
run_label_list = {'b3'}; % dimensions: (1,n_case), compact labels.



box_zc = {run_Exp2_FIR_final_m500_zc50_train,Matlab_grey,...
        run_Exp2_150_final_h8_ad01_m500_b3_nonoise_pas_ite4_nobptt_l220_l2NN5_firParallOn_ep20_zc50_neoit_s13_train};
box_label_list ={'FIR','Grey-box','NFIR'};


%% Check passivity --- passed 

Ts_passivity = 0.005; % scalar, sampling time in seconds


linear_FIR = filt(run_Exp2_150_final_h8_ad01_m500_b3_nonoise_pas_ite4_nobptt_l220_l2NN5_firParallOn_ep20_zc50_neoit_s23_train.g_linear_m,1,Ts_passivity);
[gg1,gg2] = isPassive(linear_FIR); % gg1= 1, gg2 = 0.9995

data_list = {run_Exp2_150_final_h8_ad01_m500_b3_nonoise_pas_ite4_nobptt_l220_l2NN5_firParallOn_ep20_zc50_neoit_s13_train};


label_list = {"s13"};

n_step2_cases = numel(data_list); % scalar, number of loaded Step2 results to check
passivity_label = strings(n_step2_cases, 1); % shape (n_step2_cases, 1)
passivity_n_branch = zeros(n_step2_cases, 1); % shape (n_step2_cases, 1)
passivity_n_passive = zeros(n_step2_cases, 1); % shape (n_step2_cases, 1)
passivity_max_R = zeros(n_step2_cases, 1); % shape (n_step2_cases, 1)
passivity_max_R_branch = zeros(n_step2_cases, 1); % shape (n_step2_cases, 1)
passivity_max_violation = zeros(n_step2_cases, 1); % shape (n_step2_cases, 1)
passivity_bad_branches = strings(n_step2_cases, 1); % shape (n_step2_cases, 1)

fprintf('\nFIR passivity summary for loaded Step2 banks using isPassive, Ts = %.6g sec:\n', Ts_passivity);
for i_case = 1:n_step2_cases
    label_text = label_list{i_case}; % scalar text label for this Step2 result
    D_step2 = data_list{i_case}; % scalar struct, one loaded Step2 result
    [pf_j, R_j] = func_fir_bank_passivity(D_step2.g_bank, Ts_passivity);

    func_print_passivity_summary(label_text, pf_j, R_j);

    [n_branch, n_passive, max_R, max_R_branch, max_violation, bad_branch_text] = ...
        func_passivity_summary_values(pf_j, R_j);
    passivity_label(i_case, 1) = string(label_text);
    passivity_n_branch(i_case, 1) = n_branch;
    passivity_n_passive(i_case, 1) = n_passive;
    passivity_max_R(i_case, 1) = max_R;
    passivity_max_R_branch(i_case, 1) = max_R_branch;
    passivity_max_violation(i_case, 1) = max_violation;
    passivity_bad_branches(i_case, 1) = string(bad_branch_text);
end

passivity_summary_table = table(passivity_label, passivity_n_branch, passivity_n_passive, ...
    passivity_max_R, passivity_max_R_branch, passivity_max_violation, passivity_bad_branches, ...
    'VariableNames', {'label', 'n_branch', 'n_passive', 'max_R', ...
                      'max_R_branch', 'max_violation', 'bad_branches'});
disp(passivity_summary_table);


%% Get Tabel II values
n_sample_out = 50;

box_D_list = box_zc;
n_run_box = numel(box_D_list);   
box_label_plot_list = box_label_list(1:n_run_box); % dimensions: (1,n_run_box), labels for loaded box plot runs.



fit_train_open = cell(1, n_run_box);
fit_train_cl = cell(1, n_run_box);

fit_val_open = cell(1, n_run_box);
fit_val_cl = cell(1, n_run_box);

fit_test_open = cell(1, n_run_box);
fit_test_cl = cell(1, n_run_box);




for i_run_box = 1:n_run_box
    D_box = box_D_list{i_run_box};

    [fit_train_open{i_run_box},~] = helper_func_new_metrics_by_traj( ...
        D_box.y_train_batch(n_sample_out:end,:), D_box.y_pre_train_batch(n_sample_out:end,:));
    [fit_train_cl{i_run_box},~] = helper_func_new_metrics_by_traj( ...
        D_box.y_train_batch(n_sample_out:end,:), D_box.y_pre_train_batch_cl(n_sample_out:end,:));

    [fit_val_open{i_run_box},~] = helper_func_new_metrics_by_traj( ...
        D_box.y_val_batch(n_sample_out:end,:), D_box.y_pre_val_batch(n_sample_out:end,:));
    [fit_val_cl{i_run_box},~] = helper_func_new_metrics_by_traj( ...
        D_box.y_val_batch(n_sample_out:end,:), D_box.y_pre_val_batch_cl(n_sample_out:end,:));

    [fit_test_open{i_run_box},~] = helper_func_new_metrics_by_traj( ...
        D_box.y_test_batch(n_sample_out:end,:), D_box.y_pre_test_batch(n_sample_out:end,:));
    [fit_test_cl{i_run_box},~] = helper_func_new_metrics_by_traj( ...
        D_box.y_test_batch(n_sample_out:end,:), D_box.y_pre_test_batch_cl(n_sample_out:end,:));
end


% Plot box 
figure('Name', 'ue1 fitting', 'Color', 'w');
% ax_train_fit = figure;
local_plot_closed_loop_box(fit_train_cl, box_label_plot_list, ...
    'Fitting (%)', 'ue1 fitting');

% Validation-based early stopping is not used in this Exp2 NFIR run, so uv1
% remains unseen during training.
figure('Name', 'uv1 fitting', 'Color', 'w');
% ax_test_fit = figure;
local_plot_closed_loop_box(fit_val_cl, box_label_plot_list, ...
    'Fitting (%)', 'uv1 fitting');

figure('Name', 'uv2 and uv3 fitting', 'Color', 'w');
% ax_test_fit = figure;
local_plot_closed_loop_box(fit_test_cl, box_label_plot_list, ...
    'Fitting (%)', 'uv2 and uv3 fitting');





function local_plot_closed_loop_box(data_cl, run_labels, ylab_text, title_text)
    % data_cl: (1, nRun) cell, each cell contains (B_i,1) closed-loop values
    % run_labels: (1, nRun), label cell array

    n_run_local = numel(data_cl); % scalar, how many models we are testing

    if numel(run_labels) ~= n_run_local
        error('Boxplot data/label size mismatch.');
    end
    
    x_center = 1:n_run_local; % (1,nRun)

    set(gca, 'NextPlot', 'add');
    for i_run_local = 1:n_run_local
        boxplot(data_cl{i_run_local}, 'Positions', x_center(i_run_local), 'Widths', 0.38, ...
            'Colors', [0.8500, 0.3250, 0.0980], 'Symbol', 'k+');
    end

    xlim([0.4, n_run_local + 0.6]);
    xticks(x_center);
    xticklabels(run_labels);
    ylabel(ylab_text);
    title(title_text);
    grid on;

    h_cl = plot(nan, nan, '-', 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 1.5);
    legend(h_cl, {'closed-loop'}, 'Location', 'best');
    set(gca, 'NextPlot', 'replace');

end
