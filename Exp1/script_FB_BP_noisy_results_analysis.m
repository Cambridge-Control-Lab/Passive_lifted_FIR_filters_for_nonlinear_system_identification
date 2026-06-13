clear variables;
close all;
clc;

if isfolder('Results')
    exp1_dir = pwd; % dimensions: (1,n_char), Exp1 folder.
elseif isfolder(fullfile('Exp1', 'Results'))
    exp1_dir = fullfile(pwd, 'Exp1'); % dimensions: (1,n_char), Exp1 folder.
else
    error('Run this script from the repo root or from the Exp1 folder.');
end

addpath(fullfile(exp1_dir, 'utility_functions'));

result_file_nonoise = fullfile(exp1_dir, 'Results', ...
    'run_NFIR_0_h4_ad01_m50_b10_nonoise_pas_ite5_lastbptt_l21_neoit_s14_train.mat'); % dimensions: (1,n_char), noise-free result MAT path.
result_file_noise = fullfile(exp1_dir, 'Results', ...
    'run_NFIR_0_h4_ad01_m50_b10_noise_pas_ite5_lastbptt_l21_neoit_s14_train.mat'); % dimensions: (1,n_char), noisy result MAT path.

load(result_file_nonoise);
load(result_file_noise);

D_nonoise = run_NFIR_0_h4_ad01_m50_b10_nonoise_pas_ite5_lastbptt_l21_neoit_s14_train; % dimensions: scalar struct, noise-free FB-BP result.
D_noise = run_NFIR_0_h4_ad01_m50_b10_noise_pas_ite5_lastbptt_l21_neoit_s14_train; % dimensions: scalar struct, noisy FB-BP result.

% This script plots the FB-BP box in the right panel of Fig. 4.
%
% Paper metric:
% Fit[%] = 100 * (1 - ||y - y_hat||_2 / ||y||_2).
%
% In this metric, y is the reference output trajectory used as ground truth,
% and y_hat is the model-predicted output trajectory.
%
% For the noisy experiment, Gaussian white noise with SNR = 10 dB is added
% to the output during training. The input signals are noise-free in both the
% training and test datasets; no noise is added to the inputs. The output
% reference y in the Fit metric is still the clean ground-truth output. When
% reporting Fit[%], we compare the prediction from the model trained with
% noisy output data against this clean y. This checks whether the model can
% recover the clean output despite noisy training outputs.
%
% Therefore:
% D_nonoise.y_test_batch is y, the clean ground-truth test output.
% D_noise.y_pre_test_batch_cl is y_hat, the closed-loop prediction from the
% model trained with noisy output data.
[fit_test_cl, ~] = func_new_metrics_by_traj( ...
    D_nonoise.y_test_batch, D_noise.y_pre_test_batch_cl); % dimensions: (n_test_traj,1), noisy prediction fit against clean ground truth.

figure;
boxplot(fit_test_cl);
