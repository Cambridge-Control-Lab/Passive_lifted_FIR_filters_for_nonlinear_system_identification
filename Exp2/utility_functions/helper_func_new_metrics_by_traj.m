% function [outputArg1,outputArg2] = untitled2(inputArg1,inputArg2)
% %UNTITLED2 Summary of this function goes here
% %   Detailed explanation goes here
% arguments (Input)
%     inputArg1
%     inputArg2
% end
% 
% arguments (Output)
%     outputArg1
%     outputArg2
% end
% 
% outputArg1 = inputArg1;
% outputArg2 = inputArg2;
% end

function [fit_vec, mse_vec] = func_new_metrics_by_traj(y_true_mat, y_pred_mat)
    % Input dimensions:
    % y_true_mat: (T, B), true trajectory matrix
    % y_pred_mat: (T, B), predicted trajectory matrix
    % Output dimensions:
    % fit_vec: (B, 1), fitting score for each trajectory
    % mse_vec: (B, 1), MSE for each trajectory
    arguments (Input)
        y_true_mat
        y_pred_mat
    end
    arguments (Output)
        fit_vec
        mse_vec
    end

    n_batch = size(y_true_mat, 2); % scalar B
    
    fit_vec = zeros(n_batch, 1);
    mse_vec = zeros(n_batch, 1);

    for i_batch = 1:n_batch
        y_true = y_true_mat(:, i_batch); % (T,1)
        y_pred = y_pred_mat(:, i_batch); % (T,1)
    
        err_vec = y_true - y_pred; % (T,1)

        % fit_vec(i_batch, 1) = 100 * (1 - goodnessOfFit(y_pred, y_true, 'NRMSE'));
        fit_vec(i_batch, 1) = 100 * (1 - norm(err_vec, 2) / norm(y_true, 2));
        r = rmse(y_true, y_pred); % scalar
        mse_vec(i_batch, 1) = r^2;
    end
end
