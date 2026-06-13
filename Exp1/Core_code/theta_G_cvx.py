from __future__ import annotations
"""
Notation:
    - T: number of time samples
    - B: number of trajectories
    - J: number of branches
    - M: FIR length
    - Ms: number of passivity frequency slices

    Theory links:
    - (E5) objective = train_SSE + l2reg * sum_j ||g_j||_2^2
    - (E6) decay bounds: -rho0_j * rho_j^m <= g_j(m) <= rho0_j * rho_j^m
    - (E7) passivity: V * g_j >= eps_j * 1, with V_qm = cos((q*pi/Ms)*m)
"""
from typing import Any

import cvxpy as cp
import mosek  # noqa: F401
import numpy as np

from . import theta_G_mimo


def build_passivity_matrix(m_fir: int, ms_passivity: int) -> np.ndarray:
    """
        Build sampled-frequency cosine matrix V for passivity inequality.

        Input:
        - m_fir: int, FIR length M
        - ms_passivity: int, number of frequency slices Ms

        Output:
        - v_qm: np.ndarray, shape (Ms+1, M), dtype float64
    """
    # Convert to integers.
    m_count = int(m_fir)
    ms_count = int(ms_passivity)   

    # Validate values.
    if m_count < 1:
        raise ValueError("m_fir must be >= 1")
    if ms_count < 1:
        raise ValueError("ms_passivity must be >= 1")

     # Allocate matrix.
    v_qm = np.zeros((ms_count + 1, m_count), dtype=float)

    # Fill with cosine samples.
    for q_index in range(ms_count + 1):
        for m_index in range(m_count):
            angle = (float(q_index) * np.pi / float(ms_count)) * float(m_index)
            v_qm[q_index, m_index] = np.cos(angle)

    # Return matrix.
    return v_qm


def _validate_solver_inputs(
    u_tb: np.ndarray,
    y_tb: np.ndarray,
    k_jtb: np.ndarray,
    train_idx: np.ndarray,
    m_fir: int,
    rho_j: np.ndarray,
    rho0_j: np.ndarray,
    eps_j: np.ndarray,
    ms_passivity: int,
) -> tuple[int, int, int]:
    """
    Validate solver input arrays and return resolved dimensions.

    Output:
    - (j_count, t_count, b_count): tuple[int, int, int]
    """
    # Cast arrays to expected dtypes/shapes.
    u_arr = np.asarray(u_tb, dtype=float)
    y_arr = np.asarray(y_tb, dtype=float)
    k_arr = np.asarray(k_jtb, dtype=float)
    train = np.asarray(train_idx, dtype=int).reshape(-1)
    rho = np.asarray(rho_j, dtype=float).reshape(-1)
    rho0 = np.asarray(rho0_j, dtype=float).reshape(-1)
    eps = np.asarray(eps_j, dtype=float).reshape(-1)

    # Basic shape checks.
    if u_arr.ndim != 2:
        raise ValueError("u_tb must have shape (T,B)")
    if y_arr.shape != u_arr.shape:
        raise ValueError("y_tb must have shape (T,B) and match u_tb")
    if k_arr.ndim != 3:
        raise ValueError("k_jtb must have shape (J,T,B)")

    # Resolve dimensions.
    t_count = int(u_arr.shape[0])
    b_count = int(u_arr.shape[1])
    j_count = int(k_arr.shape[0])

    # Validate cross dimensions.
    if k_arr.shape != (j_count, t_count, b_count):
        raise ValueError("k_jtb shape must be (J,T,B) aligned with u_tb")

    # Validate branch-vector lengths.
    if rho.shape[0] != j_count or rho0.shape[0] != j_count or eps.shape[0] != j_count:
        raise ValueError("rho_j, rho0_j, eps_j must have length J")

    # Validate split indices.
    if train.shape[0] < 1:
        raise ValueError("train_idx must contain at least one batch index")
    if np.any(train < 0) or np.any(train >= b_count):
        raise ValueError("train_idx contains out-of-range index")

    # Validate scalar parameters.
    if int(m_fir) < 1:
        raise ValueError("m_fir must be >= 1")
    if int(ms_passivity) < 1:
        raise ValueError("ms_passivity must be >= 1")

    # Validate decay/passivity vectors.
    if np.any(rho <= 0.0) or np.any(rho >= 1.0):
        raise ValueError("rho_j values must be in (0,1)")
    if np.any(rho0 <= 0.0):
        raise ValueError("rho0_j values must be > 0")
    if np.any(eps < 0.0):
        raise ValueError("eps_j values must be >= 0")

    # Return dimensions.
    return j_count, t_count, b_count


def build_step2_problem(
    u_tb: np.ndarray,
    y_tb: np.ndarray,
    k_jtb: np.ndarray,
    train_idx: np.ndarray,
    m_fir: int,
    l2reg: float,
    is_passive: bool,
    rho_j: np.ndarray,
    rho0_j: np.ndarray,
    eps_j: np.ndarray,
    ms_passivity: int,
)   -> tuple[cp.Problem, cp.Variable, dict[str, Any]]:
    
    """
        Build Step2 CVX problem from raw arrays.

        Input dimensions:
        - u_tb: (T,B)
        - y_tb: (T,B)
        - k_jtb: (J,T,B)
        - train_idx: (B_train,)
        - rho_j: (J,)
        - rho0_j: (J,)
        - eps_j: (J,)

        Output:
        - problem: cvxpy.Problem
        - g_jm_var: cvxpy Variable, shape (J,M)
        - aux: dict with np arrays used for diagnostics
    """
    # Validate and resolve dimensions.
    j_count, _, _ = _validate_solver_inputs(
        u_tb=u_tb,
        y_tb=y_tb,
        k_jtb=k_jtb,
        train_idx=train_idx,
        m_fir=int(m_fir),
        rho_j=rho_j,
        rho0_j=rho0_j,
        eps_j=eps_j,
        ms_passivity=int(ms_passivity),
    )

    # Cast arrays after validation.
    u_arr = np.asarray(u_tb, dtype=float)
    y_arr = np.asarray(y_tb, dtype=float)
    k_arr = np.asarray(k_jtb, dtype=float)
    train = np.asarray(train_idx, dtype=int).reshape(-1)
    rho = np.asarray(rho_j, dtype=float).reshape(-1)
    rho0 = np.asarray(rho0_j, dtype=float).reshape(-1)
    eps = np.asarray(eps_j, dtype=float).reshape(-1)

    m_count = int(m_fir)
    l2_value = float(l2reg)
    passive_flag = bool(is_passive)
    ms_count = int(ms_passivity)

    # Create opt variable 
    g_jm_var = cp.Variable((j_count, m_count)) # full MIMO matrix

    # Build Eq (E5) objective 
    cost_expr = 0.0

    # Loop over train trajectories. 
    for b_index in train:
        u_t = u_arr[:, int(b_index)] # u_arr shape (T,B), so u_t is (T,)
        y_t = y_arr[:, int(b_index)] # y_arr shape (T,B), so y_t is (T,)

        # Accumulate traj prediction
        y_hat_expr = 0.0

        # Loop over branches 
        for j_index in range(j_count):
            # Get one batch of data from this branch 
            k_t = k_arr[j_index, :, int(b_index)] # k_arr: (J,T,B), so k_t is (T,)

            s_t = k_t * u_t  # s_t (T,)
            
            phi_tm = theta_G_mimo.build_fir_regression_matrix(signal_t=s_t,m_fir=m_count)

            # FIR output from Phi(s) and branch filter coefficients g_j.
            fir_out_t = phi_tm @ g_jm_var[j_index, :] # (T,M) @ (M,) = (T,)

            # Apply final k(t) scaling explicitly: y_branch(t)=k(t)*fir_out(t).
            y_branch_t = cp.multiply(k_t, fir_out_t) # (T,)

            # Add branch contribution into total trajectory prediction.
            y_hat_expr = y_hat_expr + y_branch_t

        cost_expr = cost_expr + cp.sum_squares(y_hat_expr - y_t)

    # Add L2 regularization term.
    cost_expr = cost_expr + l2_value * cp.sum_squares(g_jm_var)

    # Build constraints list.
    constraints = []

    # Build Eq (E6) decay envelope.
    # Build time-power axis as row: shape (1,M).
    m_row = np.arange(m_count, dtype=float).reshape(1, m_count)

    # Build branch parameters as column vectors: shape (J,1).
    rho_col = rho.reshape(j_count, 1)
    rho0_col = rho0.reshape(j_count, 1)

    # Build elementwise power term: rho_power_jm[j,m] = rho_j^m, shape (J,M).
    """ Python trick
        >>> g1
        array([[1],
            [2],
            [3],
            [4]])
        >>> g2
        array([1, 2, 3, 4])
        >>> g1 ** g2
        array([[  1,   1,   1,   1],
            [  2,   4,   8,  16],
            [  3,   9,  27,  81],
            [  4,  16,  64, 256]])
        >>> g1 * (g1 ** g2)
        array([[   1,    1,    1,    1],
            [   4,    8,   16,   32],
            [   9,   27,   81,  243],
            [  16,   64,  256, 1024]])
    """
    rho_power_jm = rho_col ** m_row

    # Final decay envelope: env[j,m] = rho0_j * rho_j^m, shape (J,M).
    decay_envelope_jm = rho0_col * rho_power_jm

    # Add upper and lower bound constraints elementwise for all (j,m).
    constraints.append(g_jm_var <= decay_envelope_jm)
    constraints.append(g_jm_var >= -decay_envelope_jm)

    # Build Eq (E7) sampled passivity matrix.
    passivity_matrix_v_qm = build_passivity_matrix(m_count, ms_count)
    ones_q = np.ones(passivity_matrix_v_qm.shape[0], dtype=float)

    # Add passivity constraints when enabled.
    if passive_flag:
        for j_index in range(j_count):
            rhs_q = float(eps[j_index]) * ones_q
            constraints.append(passivity_matrix_v_qm @ g_jm_var[j_index, :] >= rhs_q)

    # Build CVX problem.
    objective = cp.Minimize(cost_expr)
    problem = cp.Problem(objective, constraints)

    # Build auxiliary dictionary.
    aux = {}
    aux["decay_envelope_jm"] = decay_envelope_jm
    aux["passivity_matrix_v_qm"] = passivity_matrix_v_qm

    # Return problem objects.
    return problem, g_jm_var, aux


def solve_step2_cvx_min(
    u_tb: np.ndarray,
    y_tb: np.ndarray,
    k_jtb: np.ndarray,
    train_idx: np.ndarray,
    m_fir: int,
    l2reg: float,
    is_passive: bool,
    rho_j: np.ndarray,
    rho0_j: np.ndarray,
    eps_j: np.ndarray,
    ms_passivity: int,
    solver_name: str,
    verbose_solver: bool,
) -> dict[str, Any]:
    """
        Solve Step2 CVX problem using MOSEK only.

        Output dictionary keys:
        - g_jm: np.ndarray, shape (J,M)
        - solver_status: str
        - opt_value: float
        - t_compile: float
        - t_solve: float
        - decay_envelope_jm: np.ndarray, shape (J,M)
        - passivity_matrix_v_qm: np.ndarray, shape (Ms+1,M)
    """
    # Enforce MOSEK-only policy.
    if str(solver_name).upper() != "MOSEK":
        raise ValueError("Step2_min solver is fixed to MOSEK")

    # Build optimization problem.
    problem, g_jm_var, aux = build_step2_problem(
        u_tb=u_tb,
        y_tb=y_tb,
        k_jtb=k_jtb,
        train_idx=train_idx,
        m_fir=int(m_fir),
        l2reg=float(l2reg),
        is_passive=bool(is_passive),
        rho_j=np.asarray(rho_j, dtype=float),
        rho0_j=np.asarray(rho0_j, dtype=float),
        eps_j=np.asarray(eps_j, dtype=float),
        ms_passivity=int(ms_passivity),
    )

    # Build solve kwargs.
    solve_kwargs = {}
    solve_kwargs["solver"] = cp.MOSEK
    solve_kwargs["verbose"] = bool(verbose_solver)
    solve_kwargs["canon_backend"] = cp.SCIPY_CANON_BACKEND

    # Solve problem.
    problem.solve(**solve_kwargs)

    # Read solver status.
    status_text = str(problem.status)
    if status_text not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Step2_min solve failed with status: {status_text}")

    # Ensure solution exists.
    if g_jm_var.value is None:
        raise RuntimeError("Step2_min solver returned no g_jm values")

    # Convert solution to numpy.
    g_jm = np.asarray(g_jm_var.value, dtype=float)

    # Gather times.
    compile_time = 0.0
    if problem.compilation_time is not None:
        compile_time = float(problem.compilation_time)

    solve_time = 0.0
    if problem.solver_stats is not None:
        if problem.solver_stats.solve_time is not None:
            solve_time = float(problem.solver_stats.solve_time)

    # Build output dictionary.
    out = {}
    out["g_jm"] = g_jm
    out["solver_status"] = status_text
    out["opt_value"] = float(problem.value)
    out["t_compile"] = compile_time
    out["t_solve"] = solve_time
    out["decay_envelope_jm"] = np.asarray(aux["decay_envelope_jm"], dtype=float)
    out["passivity_matrix_v_qm"] = np.asarray(aux["passivity_matrix_v_qm"], dtype=float)
    
    # Return solve output.
    return out
