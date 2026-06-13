function [n_branch, n_passive, max_R, max_idx, max_violation, bad_branch_text] = func_passivity_summary_values(pf_vec, R_vec)
    % pf_vec, R_vec dimensions: (J, 1)

    n_branch = numel(R_vec); % scalar J
    not_passive_idx = find(~pf_vec | R_vec > 1); % (Nbad, 1)
    [max_R, max_idx] = max(R_vec);
    max_violation = max(max_R - 1, 0);
    n_passive = n_branch - numel(not_passive_idx);

    if isempty(not_passive_idx)
        bad_branch_text = "none";
    else
        bad_branch_text = string(mat2str(reshape(not_passive_idx, 1, [])));
    end
end
