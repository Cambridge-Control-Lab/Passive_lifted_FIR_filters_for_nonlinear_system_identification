"""
theta_G_parallel_fir_mimo.py

MIMO/FIR forward helpers for Exp2 theta_G and theta_N code.

Role in the workflow:
- These routines implement the causal FIR bookkeeping used by the NFIR
  operator in arXiv:2508.05279v2 Eq. (7)-Eq. (10).
- Exp2 adds ``linear_fir_forward`` for the standalone linear FIR path used in
  the industrial-robot experiment.
- ``nfir_forward_diagonal_mimo`` evaluates branch inputs, FIR branch outputs,
  and their sum for diagnostics and rollouts.

Notation used everywhere in this file:
- T: number of time samples per trajectory
- B: number of trajectories
- J: number of branches/channels
- M: FIR length
"""
from __future__ import annotations

import numpy as np
import time

def build_fir_regression_matrix(signal_t: np.ndarray, m_fir: int) -> np.ndarray:
    """
        Build causal FIR regression matrix Phi_tm for one 1D time signal.

        Input:
        - signal_t: np.ndarray, shape (T,), dtype float64
        - m_fir: int, FIR length M

        Output:
        - phi_tm: np.ndarray, shape (T,M), dtype float64
        phi_tm[t,m] = signal_t[t-m] for t-m >= 0, else 0.
    """
    signal_1d = np.asarray(signal_t, dtype=float).reshape(-1)  # shape (T,)

    # Read dimensions as Python integers.
    t_count = int(signal_1d.shape[0])  # scalar T
    m_count = int(m_fir)  # scalar M

    # Validate FIR length.
    if m_count < 1:
        raise ValueError("m_fir must be >= 1")
    
    # Allocate output matrix 
    phi_tm = np.zeros((t_count, m_count), dtype=float)  # shape (T,M)

    # Fill each row/column using explicit loops.
    for t_index in range(t_count):
        for m_index in range(m_count):
            # Causal source index.
            src_t = t_index - m_index

            # Copy sample only when causal index is valid.
            if src_t >= 0:
                phi_tm[t_index, m_index] = signal_1d[src_t]

    # Return fully built regression matrix.
    return phi_tm

def linear_fir_forward(u_tb: np.ndarray, g_linear_m: np.ndarray) -> np.ndarray:
    """
    Apply standalone causal linear FIR to input u.

    Input:
    - u_tb: np.ndarray, shape (T,B)
    - g_linear_m: np.ndarray, shape (M,)

    Output:
    - y_linear_tb: np.ndarray, shape (T,B)
    """
    u_arr = np.asarray(u_tb, dtype=float)  # shape (T,B)
    g_arr = np.asarray(g_linear_m, dtype=float).reshape(-1)  # shape (M,)

    if u_arr.ndim != 2:
        raise ValueError("u_tb must have shape (T,B)")
    if g_arr.shape[0] < 1:
        raise ValueError("g_linear_m must have shape (M,) with M >= 1")

    t_count = int(u_arr.shape[0])  # scalar T
    b_count = int(u_arr.shape[1])  # scalar B
    m_count = int(g_arr.shape[0])  # scalar M

    y_linear_tb = np.zeros((t_count, b_count), dtype=float)  # shape (T,B)

    for b_index in range(b_count):
        u_t = u_arr[:, b_index]  # shape (T,)
        phi_tm = build_fir_regression_matrix(signal_t=u_t, m_fir=m_count)  # shape (T,M)
        y_linear_tb[:, b_index] = phi_tm @ g_arr  # shape (T,)

    return y_linear_tb

def diag_bank_to_mimo(g_jm: np.ndarray) -> np.ndarray:
    """
    Convert diagonal FIR bank g_jm to generic MIMO kernel g_oim.

    Input:
    - g_jm: np.ndarray, shape (J,M), dtype float64

    Output:
    - g_oim: np.ndarray, shape (J,J,M), dtype float64
      Only diagonal paths are non-zero.
    """
    # Convert to float array.
    g_diag = np.asarray(g_jm, dtype=float)

    # Validate shape rank.
    if g_diag.ndim != 2:
        raise ValueError("g_jm must have shape (J,M)")

    # Read dimensions.
    j_count = int(g_diag.shape[0])  # scalar J
    m_count = int(g_diag.shape[1])  # scalar M

    # Allocate generic MIMO kernel with zeros.
    g_oim = np.zeros((j_count, j_count, m_count), dtype=float)  # shape (J_out,J_in,M)

    # Fill only diagonal branch paths.
    for j_index in range(j_count):
        g_oim[j_index, j_index, :] = g_diag[j_index, :]

    # Return MIMO kernel.
    return g_oim


def mimo_conv_causal(s_jtb: np.ndarray, g_oim: np.ndarray) -> np.ndarray:
    """
    Apply causal MIMO FIR convolution.

    Input:
    - s_jtb: np.ndarray, shape (J_in,T,B), dtype float64
    - g_oim: np.ndarray, shape (J_out,J_in,M), dtype float64

    Output:
    - v_otb: np.ndarray, shape (J_out,T,B), dtype float64
    """
    # Convert inputs to float arrays.
    s_in = np.asarray(s_jtb, dtype=float)
    g_kernel = np.asarray(g_oim, dtype=float)

    # Validate input ranks.
    if s_in.ndim != 3:
        raise ValueError("s_jtb must have shape (J_in,T,B)")
    if g_kernel.ndim != 3:
        raise ValueError("g_oim must have shape (J_out,J_in,M)")

    # Read dimensions.
    j_in = int(s_in.shape[0])
    t_count = int(s_in.shape[1])
    b_count = int(s_in.shape[2])

    j_out = int(g_kernel.shape[0])
    j_in_kernel = int(g_kernel.shape[1])
    m_count = int(g_kernel.shape[2])

    # Validate channel compatibility.
    if j_in_kernel != j_in:
        raise ValueError("g_oim second axis must equal s_jtb first axis")

    # Build causal lag windows once for all channels/batches:
    # s_hist_jtbm[j,t,b,m] = s_in[j, t-m, b] if t-m>=0 else 0.
    s_pad = np.pad(s_in, ((0, 0), (m_count - 1, 0), (0, 0)), mode="constant")
    s_hist_jtbm = np.lib.stride_tricks.sliding_window_view(
        s_pad, window_shape=m_count, axis=1
    )[:, :, :, ::-1]

    # Fast path for diagonal kernel from diag_bank_to_mimo: o == i only.
    if j_out == j_in:
        offdiag_mask = ~np.eye(j_out, dtype=bool)
        if not np.any(g_kernel[offdiag_mask, :]):
            idx = np.arange(j_out, dtype=int)
            g_diag_jm = g_kernel[idx, idx, :]  # shape (J,M)
            return np.einsum("jtbm,jm->jtb", s_hist_jtbm, g_diag_jm, optimize=True)

    # Generic MIMO path.
    return np.einsum("oim,itbm->otb", g_kernel, s_hist_jtbm, optimize=True)


def nfir_forward_diagonal_mimo(
    k_jtb: np.ndarray,
    u_tb: np.ndarray,
    g_jm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run full theta_G/theta_N NFIR forward pass with a diagonal FIR bank.

    Input:
    - k_jtb: np.ndarray, shape (J,T,B), dtype float64
    - u_tb: np.ndarray, shape (T,B), dtype float64
    - g_jm: np.ndarray, shape (J,M), dtype float64

    Output:
    - y_hat_tb: np.ndarray, shape (T,B), dtype float64
    - y_branch_jtb: np.ndarray, shape (J,T,B), dtype float64
    - s_jtb: np.ndarray, shape (J,T,B), dtype float64
    - v_jtb: np.ndarray, shape (J,T,B), dtype float64
    - g_oim: np.ndarray, shape (J,J,M), dtype float64
    """
    # Convert inputs to float arrays.
    k_arr = np.asarray(k_jtb, dtype=float)
    u_arr = np.asarray(u_tb, dtype=float)
    g_arr = np.asarray(g_jm, dtype=float)

    # Validate ranks.
    if k_arr.ndim != 3:
        raise ValueError("k_jtb must have shape (J,T,B)")
    if u_arr.ndim != 2:
        raise ValueError("u_tb must have shape (T,B)")
    if g_arr.ndim != 2:
        raise ValueError("g_jm must have shape (J,M)")

    # Read dimensions.
    j_count = int(k_arr.shape[0])
    t_count = int(k_arr.shape[1])
    b_count = int(k_arr.shape[2])

    # Validate cross-dimensions.
    if u_arr.shape != (t_count, b_count):
        raise ValueError(f"u_tb shape mismatch: expected {(t_count, b_count)}, got {u_arr.shape}")
    if g_arr.shape[0] != j_count:
        raise ValueError("g_jm first axis must match J from k_jtb")

    # Eq (E1): branch-wise signals entering FIR.
    s_jtb = k_arr * u_arr[None, :, :]  # shape (J,T,B)

    # Build generic MIMO kernel from diagonal bank and run Eq (E2).
    g_oim = diag_bank_to_mimo(g_arr)  # shape (J,J,M)
    v_jtb = mimo_conv_causal(s_jtb, g_oim)  # shape (J,T,B)

    # Eq (E3): branch outputs.
    y_branch_jtb = k_arr * v_jtb  # shape (J,T,B)

    # Eq (E4): sum branch outputs.
    y_hat_tb = np.sum(y_branch_jtb, axis=0)  # shape (T,B)

    # Return full forward outputs.
    return y_hat_tb, y_branch_jtb, s_jtb, v_jtb, g_oim
