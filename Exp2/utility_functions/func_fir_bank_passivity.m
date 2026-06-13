function [pf_vec, R_vec] = func_fir_bank_passivity(g_bank_jm, Ts)
    % Input dimensions:
    % g_bank_jm: (J, M), FIR coefficient bank.
    % Ts: scalar, sampling time in seconds.
    % Output dimensions:
    % pf_vec: (J, 1), logical result returned by isPassive.
    % R_vec: (J, 1), relative passivity index. R < 1 satisfies passivity,
    %        R > 1 violates passivity.

    n_branch = size(g_bank_jm, 1); % scalar J
    pf_vec = false(n_branch, 1);
    R_vec = nan(n_branch, 1);

    for j_branch = 1:n_branch
        num_1m = reshape(g_bank_jm(j_branch, :), 1, []); % (1, M)
        % G = tf(num_1m, 1, Ts, 'Variable', 'z^-1');
        G = filt(num_1m,1,Ts);
        [pf, R] = isPassive(G);

        pf_vec(j_branch, 1) = logical(pf);
        R_vec(j_branch, 1) = max(double(R(:)));
    end
end