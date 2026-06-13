function func_print_passivity_summary(label_text, pf_vec, R_vec)
    % pf_vec, R_vec dimensions: (J, 1)

    n_branch = numel(R_vec); % scalar J
    not_passive_idx = find(~pf_vec | R_vec > 1); % (Nbad, 1)
    [max_R, max_idx] = max(R_vec);
    max_violation = max(max_R - 1, 0);

    fprintf('%s: passive %d/%d, max R = %.6g at FIR branch %d, max(R-1,0) = %.6g\n', ...
        label_text, n_branch - numel(not_passive_idx), n_branch, ...
        max_R, max_idx, max_violation);

    if isempty(not_passive_idx)
        fprintf('  non-passive FIR branches: none\n');
    else
        fprintf('  non-passive FIR branches: %s\n', mat2str(reshape(not_passive_idx, 1, [])));
    end
end