"""
theta_G_parallel_fir_core.py

theta_G FIR/filter-parameter optimization core for Exp2.

Position in the whole NFIR workflow:
- This file implements the theta_G update in Algorithm 1 of
  arXiv:2508.05279v2. In the paper, theta_G denotes the FIR/filter
  parameters of the lifted passive FIR model.
- Exp2 uses a parallel structure: a standalone linear FIR path plus the NFIR
  residual branches. This matches the industrial-robot setup discussed in
  arXiv v2 Section 6.2.
- Legacy variable names in this code may still contain ``step2``. Those names
  refer to the theta_G update and are kept unchanged for saved-result
  compatibility.

File-level flow map:
1. Build or load the lifting signal k_jtb, shape (J,T,B). Depending on
   ``k_source_mode``, k_jtb comes from a random MLP, an imported theta_N model,
   or a polynomial lifting basis.
2. Call ``theta_G_parallel_fir_cvx.solve_step2_cvx_min`` to solve the convex
   theta_G subproblem. This corresponds to arXiv v2 Eq. (15), with the
   regularized objective related to Eq. (14).
3. Enforce passive FIR structure through frequency-domain constraints related
   to arXiv v2 Eq. (16) and decay bounds related to Eq. (17).
4. Run open-loop and closed-loop rollouts with the solved linear FIR and NFIR
   branch bank, using the lifted FIR structure of arXiv v2 Eq. (7)-Eq. (10).
5. Save diagnostics and MATLAB/Python outputs that are consumed by the next
   theta_N update in the alternating loop.

Notation used throughout:
- T: number of time samples
- B: number of trajectories, including train, validation, and test batches
- J: number of NFIR branches
- M: FIR length
- F: number of feature dimensions
"""
from __future__ import annotations
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import scipy.io
import scipy.signal
import torch

from Exp2.Core_code import features
from Exp2.Core_code import io_data
from Exp2.Core_code.theta_N_parallel_fir_core import SharedMLP

from Exp2.Core_code import theta_G_parallel_fir_cvx as step2_min_cvx
from Exp2.Core_code import theta_G_parallel_fir_mimo as step2_min_mimo
import time
import copy

def compare_step2_dicts(
    data_a: dict[str, Any],
    data_b: dict[str, Any],
    keys: list[str],
    rtol: float,
    atol: float,
) -> list[str]:
    """
    Compare selected keys between two dictionaries.

    Input:
    - data_a: first dictionary
    - data_b: second dictionary
    - keys: list of keys to compare
    - rtol: relative tolerance for numeric values
    - atol: absolute tolerance for numeric values

    Output:
    - mismatch_messages: list[str]
      Empty list means all compared keys passed.
    """
    # Allocate mismatch message list.
    mismatch_messages: list[str] = []

    # Loop through keys to compare.
    for key_name in keys:
        # Check key presence in dictionary A.
        if key_name not in data_a:
            mismatch_messages.append(f"missing in A: {key_name}")
            continue

        # Check key presence in dictionary B.
        if key_name not in data_b:
            mismatch_messages.append(f"missing in B: {key_name}")
            continue

        # Convert both values to numpy arrays for unified handling.
        arr_a = np.asarray(data_a[key_name])
        arr_b = np.asarray(data_b[key_name])

        # Check shape match first.
        if arr_a.shape != arr_b.shape:
            mismatch_messages.append(f"{key_name}: shape mismatch {arr_a.shape} vs {arr_b.shape}")
            continue

        # Numeric compare with tolerances.
        if np.issubdtype(arr_a.dtype, np.number) and np.issubdtype(arr_b.dtype, np.number):
            if not np.allclose(arr_a, arr_b, rtol=rtol, atol=atol):
                max_abs = float(np.max(np.abs(arr_a - arr_b)))
                mismatch_messages.append(f"{key_name}: max_abs_error={max_abs:.3e}")
        else:
            # Non-numeric compare with exact equality.
            if not np.array_equal(arr_a, arr_b):
                mismatch_messages.append(f"{key_name}: non-numeric mismatch")

    # Return all mismatch messages.
    return mismatch_messages


def _to_branch_vector(value: Any, j_count: int, default_value: float) -> np.ndarray:
    """
        Expand scalar/None/array to branch vector.

        Input:
        - value: Any (None, scalar, or array-like)
        - j_count: int, branch count J
        - default_value: float, default scalar used when value is None

        Output:
        - vec_j: np.ndarray, shape (J,), dtype float64
    """
    # Handle None by filling with default.
    if value is None:
        return np.full(int(j_count), float(default_value), dtype=float)

    # Convert to 1D float array.
    arr = np.asarray(value, dtype=float).reshape(-1)

    # Expand scalar to length J.
    if arr.size == 1:
        return np.full(int(j_count), float(arr[0]), dtype=float)

    # Validate vector length.
    if arr.size != int(j_count):
        raise ValueError(f"Expected vector length J={j_count}, got {arr.size}")

    # Return normalized vector.
    return arr.astype(float)


def build_default_config_min(
        mode: str = "imported_nn") -> dict[str, Any]:
    """
        Build default plain-dict config for the theta_G optimization path.

        Input:
        - mode: "imported_nn" or "random_nn" or "poly_lifting"

        Output:
        - cfg: dict[str, Any]
    """
    # Normalize mode text.
    mode_text = str(mode).strip().lower()

    # Validate mode value.
    if mode_text not in ("imported_nn", "random_nn", "poly_lifting"):
        raise ValueError("mode must be imported_nn or random_nn or poly_lifting")

    # Create config dictionary.
    cfg: dict[str, Any] = {}

    # Run naming fields.
    cfg["save_full_diagnostics"] = False # whether save all details 
    cfg["run_name"] = "run_step2_min"
    cfg["out_dir"] = "out"
    cfg["schema_version"] = "nfir8e_step2_min_v1"
    cfg["mode"] = "step2_fir"

    # Input source mode and paths.
    cfg["k_source_mode"] = mode_text
    cfg["source_step1_pkl"] = None
    cfg["source_data_mat"] = None
    cfg["strict_source_match"] = True # if true, checke whether source_data_mat used in step 1 is same as here.
    cfg["rebuild_p7_from_uy"] = False  # bool scalar: if True, rebuild open-loop p_7tb from u_tb,y_tb.

    # Branch/FIR dimensions.
    cfg["n_branch"] = 13
    cfg["m_fir"] = 100

    # CVX settings.
    cfg["l2reg"] = 1.0
    cfg["is_passive"] = True
    cfg["enable_parallel_fir"] = True  # bool scalar: if False, standalone parallel FIR is forced to zero.
    cfg["zero_cost_first_n"] = 0  # scalar int, first N time samples ignored by CVX/MSE loss
    cfg["ms_passivity"] = 2000  # Num of sampling grid 
    cfg["solver_name"] = "MOSEK"
    cfg["verbose_solver"] = True
    # MILESTONE 1 TODO:
    # Add iterative settings here:
    # Meaning: iter0 is current behavior; n_refine_iter adds extra solve/simulate rounds.
    cfg["n_refine_iter"] = 0 # How many iteration for repeated reconstruction of rho. 0 for just one cvx solve 
    cfg["iter_p7_ts"] = 0.02 # when rebuilding p_7tb, what is the sampling time when building integration related things 
    cfg["iter_yhat_prev_weight"] = 1.0 # repeated CVX blend: c*y_hat_previous + (1-c)*raw_y_tb when rebuilding p_7tb
    cfg["iter_yhat_add_noise"] = False # whether add noise to y when rebuilding the y inside p_7tb,
    """NOTE!!!!!
        iter_yhat_add_noise and iter_yhat_lpf_enable works also for 
        initial p_t if rebuild_p7_from_uy = True! even n_refine_iter = 0
    """
    cfg["iter_yhat_noise_snr_db"] = 20.0 # when add noise when rebuilding p_7tb, the SNR ratio in db
    cfg["iter_yhat_noise_seed"] = 42  # when add noise when rebuilding p_7tb, the seed of gaussian noise
    cfg["iter_yhat_lpf_enable"] = False # whether add low pass filtering to y  when rebuilding the y inside p_7tb,
    cfg["iter_yhat_lpf_tau"] = 0.05 # when add low pass filtering to y, the low pass filter constant. If = 0.05 then G(s) = 1 / (0.05*s + 1), so pole at 20


    # Branchwise constraint vectors (None means fill from defaults).
    cfg["rho_j"] = None
    cfg["rho0_j"] = None
    cfg["eps_j"] = None

    # Branchwise default scalar constraints.
    cfg["rho_default"] = 0.93
    cfg["rho0_default"] = 100.0
    cfg["eps_default"] = 5e-3

    # Who u and y should be scaled in CL simulatio and iterative cvxpy training.
    cfg["fixed_uy_scale"] = True
    cfg["u_scale_fixed"] = float(22.0)
    cfg["y_scale_fixed"] = float(31.471743603975618)
    cfg["uy_scale_method"] = "divide"  # "softsign" => x/sqrt(x^2+s^2), "divide" => x/s
    cfg["u_max_after_scale"] = 100.0  # clip bound for scaled p0 channel
    cfg["y_max_after_scale"] = 100.0  # clip bound for scaled p1 channel
    

    # Feature 
    cfg["active_dims"] = (0, 1) # Also used in 
    cfg["delay_steps_by_dim"] = {0: 5, 1: 5}
    cfg["scale_io_by_20"] = False       # set to false since io are already scaled in matlab when preparing 
    cfg["feature_norm_mode"] = "none" # whether apply normalisation to extending scheduling sig/feature
                                      # Set it none, since we already apply normalisation to scheduling signal (outside this script) before they become extending scheduling signag 
    cfg["x_max"] = None  # None: no clipping; float > 0 clips normalized feature x to [-x_max, x_max].

    # random-MLP settings used only in random_nn mode.
    cfg["random_seed"] = 42
    cfg["hidden_dims"] = (128, 128)
    cfg["mlp_hidden_activation"] = "tanh"  # scalar string: hidden-layer activation for random/imported NN modes.
    cfg["mlp_output_activation"] = "tanh"  # scalar string: output activation for k_j(t) in random/imported NN modes.
    cfg["train_val_test_split"] = (16, 2, 2)
    cfg["imported_split_source"] = "step1"  # "step1": use theta_N saved split; "cfg": use theta_G train_val_test_split.
    cfg["split_seed"] = 42
    cfg["shuffle_split"] = False

    # Imported mode resolves J/M from theta_N when set to None.
    if mode_text == "imported_nn":
        cfg["n_branch"] = None
        cfg["m_fir"] = None

    # For Poly lifting mode: 0 means constant lifting only, i.e. pure FIR with k(t)=1.
    cfg["poly_order"] = int(2) # default 2nd order
    cfg["poly_basis_type"] = "legendre"  # "monomial" is current x^n basis; "legendre" uses bounded orthogonal polynomials.
    if cfg["poly_order"] not in [int(0),int(1),int(2),int(3)]:
        raise ValueError("poly_order must be in 0 1 2 3 (0 means constant lifting / pure FIR)")
    if mode_text == "poly_lifting":
        cfg["shuffle_split"] = False
        feature_map = features.build_feature_map(
            active_dims=tuple(cfg["active_dims"]),
            delay_steps=dict(cfg["delay_steps_by_dim"]),
        )
        f_count = int(feature_map.shape[0])
        poly_order = int(cfg["poly_order"])
        if poly_order == 0:
            cfg["n_branch"] = int(1)
        elif poly_order == 1:
            cfg["n_branch"] = int(f_count + 1)
        elif poly_order == 2:
            cfg["n_branch"] = int((f_count + 1) * (f_count + 2) // 2)
        else:
            cfg["n_branch"] = int((f_count + 1) * (f_count + 2) * (f_count + 3) // 6)

    # Return default config.
    return cfg


def _validate_cfg(cfg: dict[str, Any]) -> None:
    """
    Validate plain-dict theta_G config.

    Input:
    - cfg: dict[str, Any]

    Output:
    - none (raises ValueError on invalid config)
    """
    # Validate mode.
    mode_text = str(cfg.get("k_source_mode", "")).strip().lower()
    if mode_text not in ("imported_nn", "random_nn", "poly_lifting"):
        raise ValueError("k_source_mode must be imported_nn or random_nn or poly_lifting")

    # Validate solver.
    if str(cfg.get("solver_name", "")).upper() != "MOSEK":
        raise ValueError("solver_name must be MOSEK")

    # Validate output naming fields.
    if len(str(cfg.get("run_name", "")).strip()) == 0:
        raise ValueError("run_name must be non-empty")
    if len(str(cfg.get("out_dir", "")).strip()) == 0:
        raise ValueError("out_dir must be non-empty")

    # Validate mode-specific required sources.
    if mode_text == "imported_nn":
        if cfg.get("source_step1_pkl", None) is None:
            raise ValueError("imported_nn mode requires source_step1_pkl")
        # In imported mode, if a MAT path is provided it must strictly match theta_N data.
        if cfg.get("source_data_mat", None) is not None and not bool(cfg.get("strict_source_match", False)):
            raise ValueError("imported_nn mode requires strict_source_match=True when source_data_mat is provided")
    if mode_text == "random_nn" or mode_text == "poly_lifting":
        if cfg.get("source_data_mat", None) is None:
            raise ValueError("random_nn or poly_lifting mode requires source_data_mat")
        if cfg.get("n_branch", None) is None:
            raise ValueError("random_nn or poly_lifting mode requires n_branch")
        if cfg.get("m_fir", None) is None:
            raise ValueError("random_nn or poly_lifting mode requires m_fir")

    # Validate train/val/test split tuple for random mode path.
    split_tuple = tuple(cfg.get("train_val_test_split", ()))
    if len(split_tuple) != 3:
        raise ValueError("train_val_test_split must have 3 values")
    if int(split_tuple[0]) < 1 or int(split_tuple[1]) < 1 or int(split_tuple[2]) < 1:
        raise ValueError("train_val_test_split values must be >= 1")
    imported_split_source_text = str(cfg.get("imported_split_source", "step1")).strip().lower()
    if imported_split_source_text not in ("step1", "cfg"):
        raise ValueError("imported_split_source must be 'step1' or 'cfg'")

    # Validate hidden_dims shape.
    hidden_dims = tuple(cfg.get("hidden_dims", ()))
    if len(hidden_dims) != 2:
        raise ValueError("hidden_dims must contain exactly 2 values")
    if int(hidden_dims[0]) < 1 or int(hidden_dims[1]) < 1:
        raise ValueError("hidden_dims values must be >= 1")

    hidden_activation_text = str(cfg.get("mlp_hidden_activation", "tanh")).strip().lower()
    if hidden_activation_text not in ("tanh", "relu", "gelu", "sigmoid", "identity"):
        raise ValueError("mlp_hidden_activation must be tanh, relu, gelu, sigmoid, or identity")
    output_activation_text = str(cfg.get("mlp_output_activation", "tanh")).strip().lower()
    if output_activation_text not in ("tanh", "sigmoid", "identity"):
        raise ValueError("mlp_output_activation must be tanh, sigmoid, or identity")

    # Fixed scaling/split policy for current workflow.
    if bool(cfg.get("scale_io_by_20", False)):
        raise ValueError("scale_io_by_20 must be False")
    if bool(cfg.get("shuffle_split", False)):
        raise ValueError("shuffle_split must be False")
    feature_norm_mode_text = str(cfg.get("feature_norm_mode", "")).strip().lower()
    if feature_norm_mode_text not in ("none", "zscore"):
        raise ValueError("feature_norm_mode must be 'none' or 'zscore'")
    if cfg.get("x_max", None) is not None:
        x_max = float(cfg["x_max"])
        if (not np.isfinite(x_max)) or x_max <= 0.0:
            raise ValueError("x_max must be None or a positive finite scalar")

    # Validate scalar optimization settings.
    if float(cfg.get("l2reg", 0.0)) < 0.0:
        raise ValueError("l2reg must be >= 0")
    if not isinstance(cfg.get("enable_parallel_fir", True), bool):
        raise ValueError("enable_parallel_fir must be bool")
    zero_cost_first_n_raw = cfg.get("zero_cost_first_n", 0)
    if isinstance(zero_cost_first_n_raw, bool):
        raise ValueError("zero_cost_first_n must be a non-negative integer")
    if int(zero_cost_first_n_raw) != float(zero_cost_first_n_raw):
        raise ValueError("zero_cost_first_n must be an integer")
    if int(zero_cost_first_n_raw) < 0:
        raise ValueError("zero_cost_first_n must be >= 0")
    if int(cfg.get("ms_passivity", 0)) < 1:
        raise ValueError("ms_passivity must be >= 1")
    # MILESTONE 1 TODO:
    # Validate iterative settings once you add them in cfg defaults:
    # - n_refine_iter >= 0
    # - iter_p7_ts > 0
    n_refine_iter_raw = cfg.get("n_refine_iter", 0)
    if isinstance(n_refine_iter_raw, bool):
        raise ValueError("n_refine_iter must be a non-negative integer")
    if int(n_refine_iter_raw) != float(n_refine_iter_raw):
        raise ValueError("n_refine_iter must be an integer")
    if int(n_refine_iter_raw) < 0:
        raise ValueError("n_refine_iter must be >= 0")

    if float(cfg.get("iter_p7_ts",0.0)) <= 0.0:
        raise ValueError('iter_p7_ts, the sampling time,  must be >0')
    iter_yhat_prev_weight = float(cfg.get("iter_yhat_prev_weight", 1.0))
    if (not np.isfinite(iter_yhat_prev_weight)) or iter_yhat_prev_weight < 0.0 or iter_yhat_prev_weight > 1.0:
        raise ValueError("iter_yhat_prev_weight must be finite and in [0,1]")
    if not isinstance(cfg.get("iter_yhat_add_noise", False), bool):
        raise ValueError("iter_yhat_add_noise must be bool")
    if not np.isfinite(float(cfg.get("iter_yhat_noise_snr_db", 0.0))) or float(cfg.get("iter_yhat_noise_snr_db", 0.0)) <= 0.0:
        raise ValueError("iter_yhat_noise_snr_db must be finite and > 0")
    noise_seed_val = cfg.get("iter_yhat_noise_seed", None)
    if noise_seed_val is not None and isinstance(noise_seed_val, bool):
        raise ValueError("iter_yhat_noise_seed must be None or int")
    if noise_seed_val is not None:
        try:
            int(noise_seed_val)
        except Exception as exc:
            raise ValueError("iter_yhat_noise_seed must be None or int") from exc
    if not isinstance(cfg.get("iter_yhat_lpf_enable", False), bool):
        raise ValueError("iter_yhat_lpf_enable must be bool")
    if not np.isfinite(float(cfg.get("iter_yhat_lpf_tau", 0.0))) or float(cfg.get("iter_yhat_lpf_tau", 0.0)) <= 0.0:
        raise ValueError("iter_yhat_lpf_tau must be finite and > 0")

    # Validate fixed u/y scaling settings.
    if not isinstance(cfg.get("fixed_uy_scale", False), bool):
        raise ValueError("fixed_uy_scale must be bool")
    if bool(cfg.get("fixed_uy_scale", False)):
        u_scale_fixed = float(cfg.get("u_scale_fixed", 0.0))
        y_scale_fixed = float(cfg.get("y_scale_fixed", 0.0))
        if (not np.isfinite(u_scale_fixed)) or u_scale_fixed <= 0.0:
            raise ValueError("u_scale_fixed must be finite and > 0 when fixed_uy_scale=True")
        if (not np.isfinite(y_scale_fixed)) or y_scale_fixed <= 0.0:
            raise ValueError("y_scale_fixed must be finite and > 0 when fixed_uy_scale=True")
    uy_scale_method = str(cfg.get("uy_scale_method", "softsign")).strip().lower()
    if uy_scale_method not in ("softsign", "divide"):
        raise ValueError("uy_scale_method must be 'softsign' or 'divide'")
    u_max_after_scale = float(cfg.get("u_max_after_scale", 1.0))
    y_max_after_scale = float(cfg.get("y_max_after_scale", 1.0))
    if (not np.isfinite(u_max_after_scale)) or u_max_after_scale <= 0.0:
        raise ValueError("u_max_after_scale must be finite and > 0")
    if (not np.isfinite(y_max_after_scale)) or y_max_after_scale <= 0.0:
        raise ValueError("y_max_after_scale must be finite and > 0")
    if not isinstance(cfg.get("rebuild_p7_from_uy", False), bool):
        raise ValueError("rebuild_p7_from_uy must be bool")
    poly_basis_type = str(cfg.get("poly_basis_type", "monomial")).strip().lower()
    if poly_basis_type not in ("monomial", "legendre"):
        raise ValueError("poly_basis_type must be 'monomial' or 'legendre'")


def recheck_cfg_min(cfg: dict[str, Any], verbose: bool = True) -> dict[str, Any]:
    """
    Recheck and auto-correct mode-dependent config fields.

    Input:
    - cfg: config dictionary from caller
    - verbose: print correction messages when True

    Output:
    - cfg_fixed: corrected config dictionary
    """
    cfg_fixed = dict(cfg)
    mode_text = str(cfg_fixed.get("k_source_mode", "")).strip().lower()

    # Keep imported mode dimensions unresolved so they are loaded from theta_N.
    if mode_text == "imported_nn":
        if cfg_fixed.get("n_branch", None) is not None:
            if bool(verbose):
                print("[step2_min][cfg_recheck] imported_nn: force n_branch=None")
            cfg_fixed["n_branch"] = None
        if cfg_fixed.get("m_fir", None) is not None:
            if bool(verbose):
                print("[step2_min][cfg_recheck] imported_nn: force m_fir=None")
            cfg_fixed["m_fir"] = None

    # Recompute poly branch count from current feature settings.
    elif mode_text == "poly_lifting":
        feature_map = features.build_feature_map(
            active_dims=tuple(cfg_fixed["active_dims"]),
            delay_steps=dict(cfg_fixed["delay_steps_by_dim"]),
        )
        f_count = int(feature_map.shape[0])
        poly_order = int(cfg_fixed["poly_order"])
        if poly_order not in (0, 1, 2, 3):
            raise ValueError("poly_order must be in 0 1 2 3 (0 means constant lifting / pure FIR)")
        if poly_order == 0:
            n_branch_expected = int(1)
        elif poly_order == 1:
            n_branch_expected = int(f_count + 1)
        elif poly_order == 2:
            n_branch_expected = int((f_count + 1) * (f_count + 2) // 2)
        else:
            n_branch_expected = int((f_count + 1) * (f_count + 2) * (f_count + 3) // 6)

        n_branch_current = cfg_fixed.get("n_branch", None)
        if n_branch_current is None or int(n_branch_current) != int(n_branch_expected):
            if bool(verbose):
                print(
                    "[step2_min][cfg_recheck] poly_lifting: "
                    f"set n_branch={n_branch_expected} from F={f_count}, poly_order={poly_order} "
                    "(order 0 => constant lifting, k(t)=1)"
                )
            cfg_fixed["n_branch"] = int(n_branch_expected)

    # random_nn keeps caller-provided n_branch/m_fir.

    # Final validation after auto-fix.
    _validate_cfg(cfg_fixed)
    return cfg_fixed


def _validate_loaded_inputs(inputs: dict[str, Any]) -> None:
    """
    Validate loaded input tensor dimensions.

    Input:
    - inputs: dict with keys u_tb, y_tb, p_7tb, k_jtb, split indices, feature metadata

    Output:
    - none (raises ValueError when any shape contract fails)
    """
    # Read arrays and cast.
    u_tb = np.asarray(inputs["u_tb"], dtype=float)
    y_tb = np.asarray(inputs["y_tb"], dtype=float)
    p_7tb = np.asarray(inputs["p_7tb"], dtype=float)
    k_jtb = np.asarray(inputs["k_jtb"], dtype=float)

    # Validate core ranks and shapes.
    if u_tb.ndim != 2:
        raise ValueError("u_tb must have shape (T,B)")
    if y_tb.shape != u_tb.shape:
        raise ValueError("y_tb must match u_tb shape (T,B)")
    if p_7tb.shape != (7, u_tb.shape[0], u_tb.shape[1]):
        raise ValueError("p_7tb must have shape (7,T,B)")
    if k_jtb.ndim != 3:
        raise ValueError("k_jtb must have shape (J,T,B)")
    if k_jtb.shape[1] != u_tb.shape[0] or k_jtb.shape[2] != u_tb.shape[1]:
        raise ValueError("k_jtb must align with u_tb on (T,B)")

    # Validate split vectors.
    tr = np.asarray(inputs["split_train_idx"], dtype=int).reshape(-1)
    va = np.asarray(inputs["split_val_idx"], dtype=int).reshape(-1)
    te = np.asarray(inputs["split_test_idx"], dtype=int).reshape(-1)
    all_idx = np.concatenate([tr, va, te], axis=0)
    b_count = int(u_tb.shape[1])

    if np.any(all_idx < 0) or np.any(all_idx >= b_count):
        raise ValueError("Split indices are out of valid range")
    if len(np.unique(all_idx)) != len(all_idx):
        raise ValueError("Split indices must be unique across train/val/test")

    # Validate feature metadata shapes.
    feature_map = np.asarray(inputs["feature_map"], dtype=int)
    feature_mean = np.asarray(inputs["feature_mean"], dtype=float).reshape(-1)
    feature_std = np.asarray(inputs["feature_std"], dtype=float).reshape(-1)

    if feature_map.ndim != 2 or feature_map.shape[1] != 2:
        raise ValueError("feature_map must have shape (F,2)")
    if feature_mean.shape[0] != feature_map.shape[0]:
        raise ValueError("feature_mean length must match feature_map rows")
    if feature_std.shape[0] != feature_map.shape[0]:
        raise ValueError("feature_std length must match feature_map rows")


def _load_pickle_dict(path: str | Path) -> dict[str, Any]:
    """
        Load pickle and require top-level object to be dictionary.

        Input:
        - path: str | Path

        Output:
        - data: dict[str, Any]
    """
    # map path to Path obj
    path_obj = Path(path)

    with path_obj.open("rb") as file_obj: # r means read, b means in bytes since pickle file are in bytes
        data = pickle.load(file_obj) 

    # Validate type: check whether data is an dictionary 
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in pickle {path_obj}, got {type(data)}")

    return data


def _require_imported_step1_keys(step1: dict[str, Any]) -> None:
    """
        Validate required keys in imported theta_N pickle.

        Input:
        - step1: dictionary loaded from a theta_N pickle

        Output:
        - none (raises ValueError if missing key)
    """
    # Required keys for imported theta_N data used by theta_G.
    required = [
        "u_tb",
        "y_tb",
        "p_7tb",
        "k_jtb",
        "split_train_idx",
        "split_val_idx",
        "split_test_idx",
        "feature_map",
        "feature_mean",
        "feature_std",
    ]

    # Collect missing keys.
    missing = []
    for key_name in required:
        if key_name not in step1:
            missing.append(key_name)

    # Raise with full missing list.
    if len(missing) > 0:
        raise ValueError(f"Step1 pickle missing keys: {missing}")


def _extract_m_fir_from_step1(step1: dict[str, Any]) -> int:
    """
        Resolve FIR length M from imported theta_N dictionary.

        Priority:
        1) step1['g_bank'].shape[1]
        2) step1['cfg']['m_fir']
        3) cfg['m_fir']

        Input:
        - step1: legacy variable name for a theta_N dictionary
        - cfg: theta_G config dictionary

        Output:
        - m_fir: int
    """
    if "g_bank" in step1:
        g_bank = np.asarray(step1["g_bank"], dtype=float)
        if g_bank.ndim == 2 and g_bank.shape[1] >= 1:
            return int(g_bank.shape[1])
        
    raise ValueError("Could not resolve m_fir from Step1 pickle via g_bank key")

def _read_step1_dims(path: str | Path) -> tuple[int, int]:
    """
        Read branch count and FIR length from theta_N pickle.

        Input:
        - path: str | Path to theta_N pickle

        Output:
        - (j_count, m_fir): tuple[int, int]
    """
    step1 = _load_pickle_dict(path) # step1 is a legacy variable name for a theta_N result dict

    # Require k_jtb for J.
    if "k_jtb" not in step1:
        raise ValueError("Step1 pickle missing k_jtb")

    # Resolve J 
    k_jtb = np.asarray(step1["k_jtb"], dtype=float)
    if k_jtb.ndim != 3:
        raise ValueError("Step1 k_jtb must have shape (J,T,B)")
    j_count = int(k_jtb.shape[0])

    # Resolve M 
    m_fir = _extract_m_fir_from_step1(step1)

    return j_count, int(m_fir)

def _strict_check_mat_matches_step1(
    step1: dict[str, Any],
    loaded: dict[str, Any],
    skip_p7_check: bool = False,
) -> None:
    """
    Ensure source MAT arrays match arrays saved in a theta_N pickle.

    Input:
    - step1: legacy variable name for a theta_N dictionary
    - loaded: dict from io_data.load_training_mat

    Output:
    - none (raises ValueError if mismatch)
    """
    # Read theta_N arrays.
    u_step1 = np.asarray(step1["u_tb"], dtype=float)
    y_step1 = np.asarray(step1["y_tb"], dtype=float)
    p_step1 = np.asarray(step1["p_7tb"], dtype=float)

    # Read MAT arrays.
    u_mat = np.asarray(loaded["u_tb"], dtype=float)
    y_mat = np.asarray(loaded["y_tb"], dtype=float)
    p_mat = np.asarray(loaded["p_7tb"], dtype=float)

    # Check shapes and values.
    if u_step1.shape != u_mat.shape:
        raise ValueError("u_tb shape mismatch between Step1 and source MAT")
    if y_step1.shape != y_mat.shape:
        raise ValueError("y_tb shape mismatch between Step1 and source MAT")
    if not bool(skip_p7_check):
        if p_step1.shape != p_mat.shape:
            raise ValueError("p_7tb shape mismatch between Step1 and source MAT")

    if not np.allclose(u_step1, u_mat, rtol=1e-10, atol=1e-10):
        raise ValueError("u_tb data mismatch between Step1 and source MAT")
    if not np.allclose(y_step1, y_mat, rtol=1e-10, atol=1e-10):
        raise ValueError("y_tb data mismatch between Step1 and source MAT")
    if not bool(skip_p7_check):
        if not np.allclose(p_step1, p_mat, rtol=1e-10, atol=1e-10):
            raise ValueError("p_7tb data mismatch between Step1 and source MAT")


def _compute_feature_norm_stats(x_train_btf: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """
        Compute feature normalization stats for random_nn or poly_lifting mode.

        Input:
        - x_train_btf: np.ndarray, shape (B_train,T,F)
        - mode: "none" or "zscore"

        Output:
        - feature_mean_f: np.ndarray, shape (F,), dtype float64
        - feature_std_f: np.ndarray, shape (F,), dtype float64
    """
    # Normalize mode text.
    mode_text = str(mode).lower()

    # Resolve feature count.
    f_count = int(x_train_btf.shape[2])

    # Mode "none": identity normalization.
    if mode_text == "none":
        feature_mean_f = np.zeros(f_count, dtype=float)
        feature_std_f = np.ones(f_count, dtype=float)
        return feature_mean_f, feature_std_f

    # Mode "zscore": mean/std from train split only.
    if mode_text == "zscore":
        x_2d = x_train_btf.reshape(-1, f_count)
        feature_mean_f = np.mean(x_2d, axis=0).astype(float)
        feature_std_f = np.std(x_2d, axis=0).astype(float)

        for f_index in range(f_count):
            if feature_std_f[f_index] < 1e-12:
                feature_std_f[f_index] = 1.0

        return feature_mean_f, feature_std_f

    # Unsupported mode.
    raise ValueError(f"Unsupported feature_norm_mode: {mode_text}")


def _load_inputs_imported(cfg: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
    """
        Load inputs for imported_nn mode.

        Input:
        - cfg: config dict with source_step1_pkl path

        Output:
        - inputs: dict with normalized arrays/metadata
        - j_count: int, J
        - m_fir: int, M
    """
    # Read source path.
    step1_path = Path(str(cfg["source_step1_pkl"]))

    # Load theta_N dictionary.
    step1 = _load_pickle_dict(step1_path)

    # Validate required theta_N keys.
    _require_imported_step1_keys(step1)

    # Optional strict source MAT alignment check.
    source_data_mat = cfg.get("source_data_mat", None)
    if source_data_mat is not None and bool(cfg.get("strict_source_match", False)):
        loaded = io_data.load_training_mat(source_data_mat)
        _strict_check_mat_matches_step1(
            step1=step1,
            loaded=loaded,
            skip_p7_check=bool(cfg.get("rebuild_p7_from_uy", False)),
        )

    # Resolve J and M.
    k_jtb = np.asarray(step1["k_jtb"], dtype=float) # load nonlinearity output data
    j_count = int(k_jtb.shape[0]) # jtb so j is at shape[0]
    m_fir = _extract_m_fir_from_step1(step1)

    # Build normalized input dictionary.
    inputs: dict[str, Any] = {}
    inputs["u_tb"] = np.asarray(step1["u_tb"], dtype=float)
    inputs["y_tb"] = np.asarray(step1["y_tb"], dtype=float)
    inputs["p_7tb"] = np.asarray(step1["p_7tb"], dtype=float)
    inputs["k_jtb"] = np.asarray(step1["k_jtb"], dtype=float)

    # Imported split policy:
    # - "step1": reuse theta_N saved indices.
    # - "cfg": rebuild indices from theta_G train_val_test_split.
    imported_split_source_text = str(cfg.get("imported_split_source", "step1")).strip().lower()
    if imported_split_source_text == "cfg":
        n_batch_total = int(inputs["u_tb"].shape[1])  # scalar B, total number of trajectories
        split_train_idx, split_val_idx, split_test_idx = features.split_batch_indices(
            n_batch=n_batch_total,
            split_counts=tuple(cfg["train_val_test_split"]),
            split_seed=int(cfg["split_seed"]),
            shuffle=bool(cfg["shuffle_split"]),
        )
    else:
        split_train_idx = np.asarray(step1["split_train_idx"], dtype=int).reshape(-1)  # shape (B_train,)
        split_val_idx = np.asarray(step1["split_val_idx"], dtype=int).reshape(-1)      # shape (B_val,)
        split_test_idx = np.asarray(step1["split_test_idx"], dtype=int).reshape(-1)    # shape (B_test,)

    inputs["split_train_idx"] = np.asarray(split_train_idx, dtype=int).reshape(-1)
    inputs["split_val_idx"] = np.asarray(split_val_idx, dtype=int).reshape(-1)
    inputs["split_test_idx"] = np.asarray(split_test_idx, dtype=int).reshape(-1)

    inputs["feature_map"] = np.asarray(step1["feature_map"], dtype=int)
    inputs["feature_mean"] = np.asarray(step1["feature_mean"], dtype=float).reshape(-1)
    inputs["feature_std"] = np.asarray(step1["feature_std"], dtype=float).reshape(-1)
    # Imported NN must use the same feature clipping as theta_N training.
    # effective_x_max: None or scalar float, applied after feature normalization.
    step1_x_max = step1.get("cfg", {}).get("x_max", step1.get("x_max", None))
    step2_x_max = cfg.get("x_max", None)
    if step2_x_max is None:
        effective_x_max = step1_x_max
    elif step1_x_max is None:
        raise ValueError("imported_nn x_max mismatch: Step1 used None but Step2 requested clipping")
    elif not np.isclose(float(step1_x_max), float(step2_x_max), rtol=0.0, atol=1e-12):
        raise ValueError(
            f"imported_nn x_max mismatch: Step1 used {step1_x_max}, Step2 requested {step2_x_max}"
        )
    else:
        effective_x_max = float(step2_x_max)
    inputs["x_max"] = None if effective_x_max is None else float(effective_x_max)
    # MILESTONE 3 TODO:
    # Store NN metadata/state for iterative k_jtb rebuild:
    # - inputs["mlp_hidden_dims"] (from step1["cfg"]["hidden_dims"])
    # - inputs["mlp_n_branch"] (J)
    # - inputs["mlp_model_state_dict"] (from step1["model_state_dict"])
    inputs["mlp_hidden_dims"] = np.asarray(step1["cfg"]["hidden_dims"],dtype=int)
    inputs["mlp_hidden_activation"] = str(step1["cfg"].get("mlp_hidden_activation", "tanh"))
    inputs["mlp_output_activation"] = str(step1["cfg"].get("mlp_output_activation", "tanh"))
    inputs["mlp_n_branch"] = j_count
    inputs["mlp_model_state_dict"] = step1["model_state_dict"] # NN parameters 

    inputs["source_step1_path"] = str(step1_path)
    if source_data_mat is None:
        inputs["source_data_mat_path"] = None
    else:
        inputs["source_data_mat_path"] = str(source_data_mat)
    inputs["k_source_mode"] = "imported_nn"

    # Optional open-loop p_7tb rebuild from u_tb,y_tb for imported mode.
    if bool(cfg.get("rebuild_p7_from_uy", False)):
        p_7tb_new = _p_7tb_regen(
            u_tb=np.asarray(inputs["u_tb"], dtype=float),
            y_hat_tb=np.asarray(inputs["y_tb"], dtype=float),
            ts=float(cfg["iter_p7_ts"]),
            add_noise=bool(cfg.get("iter_yhat_add_noise", False)),
            noise_snr_db=float(cfg.get("iter_yhat_noise_snr_db", 20.0)),
            noise_seed=cfg.get("iter_yhat_noise_seed", None),
            lpf_enable=bool(cfg.get("iter_yhat_lpf_enable", False)),
            lpf_tau=float(cfg.get("iter_yhat_lpf_tau", 0.05)),
            cfg_local=cfg,
        )
        inputs["p_7tb"] = p_7tb_new
        cfg_rebuild = dict(cfg)
        cfg_rebuild["n_branch"] = int(j_count)
        cfg_rebuild["hidden_dims"] = tuple(
            int(v) for v in np.asarray(inputs["mlp_hidden_dims"], dtype=int).reshape(-1)
        )
        inputs["k_jtb"] = _rebuild_k_jtb_all_mode(
            inputs=inputs,
            cfg_local=cfg_rebuild,
            p_7tb_new=p_7tb_new,
        )

    # Validate loaded input contracts.
    _validate_loaded_inputs(inputs)

    # Return inputs and dimensions.
    return inputs, j_count, int(m_fir)


def _load_inputs_random(cfg: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
    """
        Load inputs for random_nn mode.

        Input:
        - cfg: config dict with random_nn settings

        Output:
        - inputs: dict with normalized arrays/metadata
        - j_count: int, J
        - m_fir: int, M
    """
    
    # Read and validate required scalars.
    source_data_mat = str(cfg["source_data_mat"])
    j_count = int(cfg["n_branch"])
    m_fir = int(cfg["m_fir"])

    if j_count < 1:
        raise ValueError("n_branch must be >= 1")
    if m_fir < 1:
        raise ValueError("m_fir must be >= 1")

    # Load MAT data.
    loaded = io_data.load_training_mat(source_data_mat)
    u_tb = np.asarray(loaded["u_tb"], dtype=float)
    y_tb = np.asarray(loaded["y_tb"], dtype=float)
    p_7tb = np.asarray(loaded["p_7tb"], dtype=float)
    if bool(cfg.get("rebuild_p7_from_uy", False)):
        p_7tb = _p_7tb_regen(
            u_tb=u_tb,
            y_hat_tb=y_tb,
            ts=float(cfg["iter_p7_ts"]),
            add_noise=bool(cfg.get("iter_yhat_add_noise", False)),
            noise_snr_db=float(cfg.get("iter_yhat_noise_snr_db", 20.0)),
            noise_seed=cfg.get("iter_yhat_noise_seed", None),
            lpf_enable=bool(cfg.get("iter_yhat_lpf_enable", False)),
            lpf_tau=float(cfg.get("iter_yhat_lpf_tau", 0.05)),
            cfg_local=cfg,
        )

    # Split on batch axis.
    n_batch = int(u_tb.shape[1])
    split_train_idx, split_val_idx, split_test_idx = features.split_batch_indices(
        n_batch=n_batch,
        split_counts=tuple(cfg["train_val_test_split"]),
        split_seed=int(cfg["split_seed"]),
        shuffle=bool(cfg["shuffle_split"]),
    )

    # Build feature map.
    feature_map = features.build_feature_map(
        active_dims=tuple(cfg["active_dims"]),
        delay_steps=dict(cfg["delay_steps_by_dim"]),
    )

    # Build extended feature tensor (F,T,B).
    p_ext_ftb = features.build_p_ext_from_p7(
        p_7tb=p_7tb,
        feature_map=feature_map,
        scale_io_by_20=bool(cfg["scale_io_by_20"]),
    )

    # Convert to model input format (B,T,F), float32 to match theta_N/prior theta_G path.
    x_all_btf = np.transpose(p_ext_ftb, (2, 1, 0)).astype(np.float32)

    # Compute feature normalization stats from train split only.
    x_train_btf = x_all_btf[split_train_idx, :, :]
    feature_mean_f, feature_std_f = _compute_feature_norm_stats(
        x_train_btf=x_train_btf,
        mode=str(cfg["feature_norm_mode"]),
    )

    # Apply normalization to all batches.
    x_all_btf = (
        x_all_btf
        - feature_mean_f[None, None, :].astype(np.float32)
    ) / feature_std_f[None, None, :].astype(np.float32)
    # x_all_btf: shape (B,T,F), normalized feature data used by fixed random NN.
    if cfg.get("x_max", None) is not None:
        x_max = float(cfg["x_max"])  # scalar clip bound for normalized feature vector x
        x_all_btf = np.clip(x_all_btf, -x_max, x_max)

    # Set deterministic seeds.
    np.random.seed(int(cfg["random_seed"]))
    torch.manual_seed(int(cfg["random_seed"]))

    # Build untrained MLP model.
    input_dim = int(x_all_btf.shape[2])
    model = SharedMLP(
        input_dim=input_dim,
        n_branch=j_count,
        hidden_dims=tuple(cfg["hidden_dims"]),
        hidden_activation=str(cfg.get("mlp_hidden_activation", "tanh")),
        output_activation=str(cfg.get("mlp_output_activation", "tanh")),
    )

    # Run model once to get fixed k values.
    x_t = torch.tensor(x_all_btf, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        k_btj_t = model(x_t)

    # Convert to numpy and reorder to (J,T,B).
    k_btj = k_btj_t.detach().cpu().numpy().astype(float)
    k_jtb = np.transpose(k_btj, (2, 1, 0)).astype(float)

    # Build normalized input dictionary.
    inputs: dict[str, Any] = {}
    inputs["u_tb"] = u_tb
    inputs["y_tb"] = y_tb
    inputs["p_7tb"] = p_7tb
    inputs["k_jtb"] = k_jtb

    inputs["split_train_idx"] = np.asarray(split_train_idx, dtype=int)
    inputs["split_val_idx"] = np.asarray(split_val_idx, dtype=int)
    inputs["split_test_idx"] = np.asarray(split_test_idx, dtype=int)

    inputs["feature_map"] = np.asarray(feature_map, dtype=int)
    inputs["feature_mean"] = np.asarray(feature_mean_f, dtype=float)
    inputs["feature_std"] = np.asarray(feature_std_f, dtype=float)
    inputs["x_max"] = cfg.get("x_max", None)
    # MILESTONE 3 TODO:
    # Store NN metadata/state for iterative k_jtb rebuild:
    # - inputs["mlp_hidden_dims"] = cfg["hidden_dims"]
    # - inputs["mlp_n_branch"] = j_count
    # - inputs["mlp_model_state_dict"] = model.state_dict() copy
    inputs["mlp_hidden_dims"] = cfg["hidden_dims"]
    inputs["mlp_hidden_activation"] = str(cfg.get("mlp_hidden_activation", "tanh"))
    inputs["mlp_output_activation"] = str(cfg.get("mlp_output_activation", "tanh"))
    inputs["mlp_n_branch"] = j_count
    inputs["mlp_model_state_dict"] = copy.deepcopy(model.state_dict())

    inputs["source_step1_path"] = None
    inputs["source_data_mat_path"] = str(source_data_mat)
    inputs["k_source_mode"] = "random_nn"

    # Validate loaded input contracts.
    _validate_loaded_inputs(inputs)

    # Return inputs and dimensions.
    return inputs, int(j_count), int(m_fir)

def _build_poly_k_jtb_from_x_all(
    x_all_btf: np.ndarray,
    poly_order: int,
    j_count: int,
    poly_basis_type: str = "monomial",
) -> np.ndarray:
    """
    Build polynomial lifting values for the theta_G update.

    Inputs:
    - x_all_btf: ndarray, shape (B,T,F), normalized feature tensor.
    - poly_order: scalar int, maximum polynomial degree.
    - j_count: scalar int, number of basis terms / branches J.
    - poly_basis_type: scalar str, either "monomial" or "legendre".

    Output:
    - k_jtb: ndarray, shape (J,T,B), lifting values used by theta_G.

    This deterministic polynomial lifting is related to arXiv:2508.05279v2
    Eq. (8). In Exp2, poly_order=0 is used for the FIR baseline and gives the
    constant lifting k(t)=1.
    """
    b_count = int(x_all_btf.shape[0])
    t_count = int(x_all_btf.shape[1])
    f_count = int(x_all_btf.shape[2])
    basis_type_text = str(poly_basis_type).strip().lower()
    if basis_type_text not in ("monomial", "legendre"):
        raise ValueError("poly_basis_type must be 'monomial' or 'legendre'")

    # k_btj has shape (B,T,J): for each batch/time, store all polynomial basis terms.
    # term_idx is the branch index j (the position of each basis term in J).
    k_btj = np.empty((b_count, t_count, j_count), dtype=np.float32)
    term_idx = 0
    # Degree-0 term: constant 1. For poly_order=0 this is the only term, so k(t)=1 (pure FIR).
    k_btj[:, :, term_idx] = 1.0
    term_idx += 1

    if basis_type_text == "monomial":
        if poly_order >= 1:
            # Degree-1 terms in order: x0, x1, ..., x(F-1).
            for i_idx in range(f_count):
                k_btj[:, :, term_idx] = x_all_btf[:, :, i_idx]
                term_idx += 1

        if poly_order >= 2:
            # Degree-2 terms are ordered to match uty10SPb:
            # 1) squares first: x0^2, x1^2, ..., x(F-1)^2
            for i_idx in range(f_count):
                # xi_bt shape: (B,T)
                xi_bt = x_all_btf[:, :, i_idx]
                k_btj[:, :, term_idx] = xi_bt * xi_bt
                term_idx += 1
            # 2) then cross terms with i<j: x0*x1, x0*x2, ..., x(F-2)*x(F-1)
            for i_idx in range(f_count):
                xi_bt = x_all_btf[:, :, i_idx]
                for j_idx in range(i_idx + 1, f_count):
                    k_btj[:, :, term_idx] = xi_bt * x_all_btf[:, :, j_idx]
                    term_idx += 1

        if poly_order >= 3:
            # Degree-3 terms in combinations-with-replacement order i<=j<=k:
            # x_i * x_j * x_k, scanned lexicographically by (i,j,k).
            for i_idx in range(f_count):
                xi_bt = x_all_btf[:, :, i_idx]
                for j_idx in range(i_idx, f_count):
                    # xij_bt shape: (B,T), reused for speed in inner k-loop.
                    xij_bt = xi_bt * x_all_btf[:, :, j_idx]
                    for k_idx in range(j_idx, f_count):
                        k_btj[:, :, term_idx] = xij_bt * x_all_btf[:, :, k_idx]
                        term_idx += 1

    else:
        # Legendre basis replaces repeated powers of one variable by:
        # raw [1, x, x^2] -> Legendre [1, x, (3*x^2 - 1)/2].
        # Cross terms still multiply first-order terms from different variables.
        l1_btf = x_all_btf
        l2_btf = 0.5 * (3.0 * x_all_btf * x_all_btf - 1.0)
        l3_btf = 0.5 * (5.0 * x_all_btf * x_all_btf * x_all_btf - 3.0 * x_all_btf)

        if poly_order >= 1:
            for i_idx in range(f_count):
                k_btj[:, :, term_idx] = l1_btf[:, :, i_idx]
                term_idx += 1

        if poly_order >= 2:
            for i_idx in range(f_count):
                k_btj[:, :, term_idx] = l2_btf[:, :, i_idx]
                term_idx += 1
            for i_idx in range(f_count):
                li_bt = l1_btf[:, :, i_idx]
                for j_idx in range(i_idx + 1, f_count):
                    k_btj[:, :, term_idx] = li_bt * l1_btf[:, :, j_idx]
                    term_idx += 1

        if poly_order >= 3:
            for i_idx in range(f_count):
                for j_idx in range(i_idx, f_count):
                    for k_idx in range(j_idx, f_count):
                        if i_idx == j_idx and j_idx == k_idx:
                            k_btj[:, :, term_idx] = l3_btf[:, :, i_idx]
                        elif i_idx == j_idx:
                            k_btj[:, :, term_idx] = l2_btf[:, :, i_idx] * l1_btf[:, :, k_idx]
                        elif j_idx == k_idx:
                            k_btj[:, :, term_idx] = l1_btf[:, :, i_idx] * l2_btf[:, :, j_idx]
                        else:
                            k_btj[:, :, term_idx] = (
                                l1_btf[:, :, i_idx]
                                * l1_btf[:, :, j_idx]
                                * l1_btf[:, :, k_idx]
                            )
                        term_idx += 1

    if term_idx != j_count:
        raise ValueError(f"poly basis branch count mismatch: generated {term_idx}, expected {j_count}")

    k_jtb = np.transpose(k_btj, (2, 1, 0)).astype(float)

    return k_jtb

def _load_inputs_polylift(cfg: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
    """
        Load inputs for random_nn mode.

        Input:
        - cfg: config dict with random_nn settings

        Output:
        - inputs: dict with normalized arrays/metadata
        - j_count: int, J
        - m_fir: int, M
    """
    
    # Read and validate required scalars.
    source_data_mat = str(cfg["source_data_mat"]) # matlab file storing all the training data
    j_count = int(cfg["n_branch"])
    m_fir = int(cfg["m_fir"])

    if j_count < 1:
        raise ValueError("n_branch must be >= 1")
    if m_fir < 1:
        raise ValueError("m_fir must be >= 1")

    # Load MAT data.
    loaded = io_data.load_training_mat(source_data_mat)
    u_tb = np.asarray(loaded["u_tb"], dtype=float)
    y_tb = np.asarray(loaded["y_tb"], dtype=float)
    p_7tb = np.asarray(loaded["p_7tb"], dtype=float)
    if bool(cfg.get("rebuild_p7_from_uy", False)):
        p_7tb = _p_7tb_regen(
            u_tb=u_tb,
            y_hat_tb=y_tb,
            ts=float(cfg["iter_p7_ts"]),
            add_noise=bool(cfg.get("iter_yhat_add_noise", False)),
            noise_snr_db=float(cfg.get("iter_yhat_noise_snr_db", 20.0)),
            noise_seed=cfg.get("iter_yhat_noise_seed", None),
            lpf_enable=bool(cfg.get("iter_yhat_lpf_enable", False)),
            lpf_tau=float(cfg.get("iter_yhat_lpf_tau", 0.05)),
            cfg_local=cfg,
        )

    # Split on batch axis.
    n_batch = int(u_tb.shape[1])
    split_train_idx, split_val_idx, split_test_idx = features.split_batch_indices(
        n_batch=n_batch,
        split_counts=tuple(cfg["train_val_test_split"]),
        split_seed=int(cfg["split_seed"]),
        shuffle=bool(cfg["shuffle_split"]),
    )

    # Build feature map.
    feature_map = features.build_feature_map(
        active_dims=tuple(cfg["active_dims"]),
        delay_steps=dict(cfg["delay_steps_by_dim"]),
    )

    # Build extended feature tensor (F,T,B).
    p_ext_ftb = features.build_p_ext_from_p7(
        p_7tb=p_7tb,
        feature_map=feature_map,
        scale_io_by_20=bool(cfg["scale_io_by_20"]),
    )

    # Convert to model input format (B,T,F), float32 to match theta_N/prior theta_G path.
    x_all_btf = np.transpose(p_ext_ftb, (2, 1, 0)).astype(np.float32)

    # Compute feature normalization stats from train split only.
    x_train_btf = x_all_btf[split_train_idx, :, :]
    feature_mean_f, feature_std_f = _compute_feature_norm_stats(
        x_train_btf=x_train_btf,
        mode=str(cfg["feature_norm_mode"]),
    )

    # Apply normalization to all batches.
    x_all_btf = (
        x_all_btf
        - feature_mean_f[None, None, :].astype(np.float32)
    ) / feature_std_f[None, None, :].astype(np.float32)
    # x_all_btf: shape (B,T,F), normalized feature data used by polynomial lifting.
    if cfg.get("x_max", None) is not None:
        x_max = float(cfg["x_max"])  # scalar clip bound for normalized feature vector x
        x_all_btf = np.clip(x_all_btf, -x_max, x_max)

    # Build Poly basis from normalized feature data x_all_btf with shape (B,T,F).
    # MILESTONE 3 TODO:
    # Extract this polynomial basis block into a helper so the same ordering is reused
    # when rebuilding k_jtb inside iterative refinement.
    
    poly_order = int(cfg["poly_order"])
    """ The following block of code is moved to the function: _build_poly_k_jtb_from_x_all
        b_count = int(x_all_btf.shape[0])
        t_count = int(x_all_btf.shape[1])
        f_count = int(x_all_btf.shape[2])

        # k_btj has shape (B,T,J): for each batch/time, store all polynomial basis terms.
        # term_idx is the branch index j (the position of each basis term in J).
        k_btj = np.empty((b_count, t_count, j_count), dtype=np.float32)
        term_idx = 0
        # Degree-0 term: constant 1. For poly_order=0 this is the only term, so k(t)=1 (pure FIR).
        k_btj[:, :, term_idx] = 1.0
        term_idx += 1

        if poly_order >= 1:
            # Degree-1 terms in order: x0, x1, ..., x(F-1).
            for i_idx in range(f_count):
                k_btj[:, :, term_idx] = x_all_btf[:, :, i_idx]
                term_idx += 1

        if poly_order >= 2:
            # Degree-2 terms are ordered to match uty10SPb:
            # 1) squares first: x0^2, x1^2, ..., x(F-1)^2
            for i_idx in range(f_count):
                # xi_bt shape: (B,T)
                xi_bt = x_all_btf[:, :, i_idx]
                k_btj[:, :, term_idx] = xi_bt * xi_bt
                term_idx += 1
            # 2) then cross terms with i<j: x0*x1, x0*x2, ..., x(F-2)*x(F-1)
            for i_idx in range(f_count):
                xi_bt = x_all_btf[:, :, i_idx]
                for j_idx in range(i_idx + 1, f_count):
                    k_btj[:, :, term_idx] = xi_bt * x_all_btf[:, :, j_idx]
                    term_idx += 1

        if poly_order >= 3:
            # Degree-3 terms in combinations-with-replacement order i<=j<=k:
            # x_i * x_j * x_k, scanned lexicographically by (i,j,k).
            for i_idx in range(f_count):
                xi_bt = x_all_btf[:, :, i_idx]
                for j_idx in range(i_idx, f_count):
                    # xij_bt shape: (B,T), reused for speed in inner k-loop.
                    xij_bt = xi_bt * x_all_btf[:, :, j_idx]
                    for k_idx in range(j_idx, f_count):
                        k_btj[:, :, term_idx] = xij_bt * x_all_btf[:, :, k_idx]
                        term_idx += 1

        if term_idx != j_count:
            raise ValueError(f"poly basis branch count mismatch: generated {term_idx}, expected {j_count}")

        k_jtb = np.transpose(k_btj, (2, 1, 0)).astype(float)
    """
    k_jtb = _build_poly_k_jtb_from_x_all(x_all_btf=x_all_btf,
                                         poly_order=poly_order,
                                         j_count=j_count,
                                         poly_basis_type=str(cfg.get("poly_basis_type", "monomial")))


    # Build normalized input dictionary.
    inputs: dict[str, Any] = {}
    inputs["u_tb"] = u_tb
    inputs["y_tb"] = y_tb
    inputs["p_7tb"] = p_7tb
    inputs["k_jtb"] = k_jtb

    inputs["split_train_idx"] = np.asarray(split_train_idx, dtype=int)
    inputs["split_val_idx"] = np.asarray(split_val_idx, dtype=int)
    inputs["split_test_idx"] = np.asarray(split_test_idx, dtype=int)

    inputs["feature_map"] = np.asarray(feature_map, dtype=int)
    inputs["feature_mean"] = np.asarray(feature_mean_f, dtype=float)
    inputs["feature_std"] = np.asarray(feature_std_f, dtype=float)
    inputs["x_max"] = cfg.get("x_max", None)
    # MILESTONE 3 TODO:
    # Optional for key consistency across modes:
    inputs["mlp_hidden_dims"] = None
    inputs["mlp_n_branch"] = j_count
    inputs["mlp_model_state_dict"] = None

    inputs["source_step1_path"] = None
    inputs["source_data_mat_path"] = str(source_data_mat)
    inputs["k_source_mode"] = "poly_lifting"

    # Validate loaded input contracts.
    _validate_loaded_inputs(inputs)

    # Return inputs and dimensions.
    return inputs, int(j_count), int(m_fir)


def _load_inputs(cfg: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
    """
        Dispatcher for input loading by source mode.

        Input:
        - cfg: config dictionary

        Output:
        - inputs: normalized input dictionary
        - j_count: int
        - m_fir: int
    """
    # Read mode text.
    mode_text = str(cfg["k_source_mode"]).strip().lower()

    # Imported mode path.
    if mode_text == "imported_nn":
        return _load_inputs_imported(cfg)

    # Random mode path.
    if mode_text == "random_nn":
        return _load_inputs_random(cfg)
    
    # Poly-lift path. 
    if mode_text == "poly_lifting":
        return _load_inputs_polylift(cfg)

    # Unsupported mode.
    raise ValueError("k_source_mode must be imported_nn or random_nn or poly_lifting")



def _p_7tb_regen(
    u_tb: np.ndarray,
    y_hat_tb: np.ndarray,
    ts: float,
    add_noise: bool,
    noise_snr_db: float,
    noise_seed: int | None,
    lpf_enable: bool,
    lpf_tau: float,
    cfg_local: dict[str, Any]
) -> np.ndarray:
    """ MILESTONE 2 TODO (new helper near data helpers):
        one helper to regenerate p_7tb from u_tb and y_hat_tb with shape contracts:
        - Input: u_tb(T,B), y_hat_tb(T,B), ts (float scalar)
        - Note u_tb is for both input train and input valid! So is y_hat_tb
        - Output: be p_7tb shape (7,T,B)

        Note:
        - B is the NOT actual trained batches number in cvx. It is total of batches provided. 

        Undelayed channels: 
        - [u_scaled, y_scaled, sin(int_y), cos(int_y), 0, 0, tanh(y_unscaled)]
        -  Delay pass only on channels 1..6 with delay1=[0; x(1:end-1)] per batch column
    """
    # y_raw_tb: (T,B), raw predicted output before optional noise/filter processing.
    y_raw_tb = np.asarray(y_hat_tb, dtype=float)
    if y_raw_tb.ndim != 2:
        raise ValueError("y_hat_tb must have shape (T,B)")

    # y_proc_tb: (T,B), working signal after optional noise/filter preprocessing.
    y_proc_tb = y_raw_tb.copy()
    y_after_noise_tb = y_proc_tb.copy()
    y_after_lpf_tb = y_proc_tb.copy()

    # Optional Gaussian noise with per-batch SNR in dB.
    if bool(add_noise):
        # sig_pow_b: (B,), mean signal power of each batch column.
        sig_pow_b = np.mean(y_proc_tb * y_proc_tb, axis=0)
        sig_pow_b = np.maximum(sig_pow_b, 1e-12)
        # noise_pow_b: (B,), target noise power from SNR relation.
        noise_pow_b = sig_pow_b / (10.0 ** (float(noise_snr_db) / 10.0))
        # noise_std_b: (B,), Gaussian std for each batch column.
        noise_std_b = np.sqrt(noise_pow_b)
        # noise_tb: (T,B), white Gaussian noise from NumPy built-in RNG.
        rng = np.random.default_rng(None if noise_seed is None else int(noise_seed))
        noise_tb = rng.normal(loc=0.0, scale=1.0, size=y_proc_tb.shape) * noise_std_b.reshape(1, -1)
        y_proc_tb = y_proc_tb + noise_tb
        y_after_noise_tb = y_proc_tb.copy()

    # Optional low-pass filter using exact ZOH discrete pole.
    if bool(lpf_enable):
        # a_lpf: scalar, discrete pole from continuous tau and sample time Ts.
        a_lpf = float(np.exp(-float(ts) / float(lpf_tau)))
        # num_b and den_a define y[k] = a*y[k-1] + (1-a)*x[k].
        num_b = np.array([1.0 - a_lpf], dtype=float)
        den_a = np.array([1.0, -a_lpf], dtype=float)
        # zi_1b: (1,B), filter initial state chosen so first-sample output starts at input.
        zi_1b = (a_lpf * y_proc_tb[0, :]).reshape(1, -1)
        y_proc_tb, _zf_1b = scipy.signal.lfilter(
            num_b,
            den_a,
            y_proc_tb,
            axis=0,
            zi=zi_1b,
        )
        y_after_lpf_tb = y_proc_tb.copy()


    p_7tb_delay = io_data.build_p7_from_u_y(
        u_tb=np.asarray(u_tb, dtype=float),
        y_tb=np.asarray(y_proc_tb, dtype=float),
        ts=float(ts),
        fixed_uy_scale=bool(cfg_local.get("fixed_uy_scale", False)),
        u_scale_fixed=float(cfg_local.get("u_scale_fixed", 22.0)),
        y_scale_fixed=float(cfg_local.get("y_scale_fixed", 31.471743603975618)),
        uy_scale_method=str(cfg_local.get("uy_scale_method", "softsign")),
        u_max_after_scale=float(cfg_local.get("u_max_after_scale", 1.0)),
        y_max_after_scale=float(cfg_local.get("y_max_after_scale", 1.0)),
    )

    return p_7tb_delay


def _rebuild_k_jtb_all_mode(inputs:dict[str, Any], cfg_local:dict[str, Any], p_7tb_new: np.ndarray) -> np.ndarray:
    """ MILESTONE 3 TODO (new helper near data helpers):
        - Add one mode-dispatch helper to rebuild k_jtb from regenerated p_7tb:
        - Input: inputs, cfg_local, p_7tb_new
        - Poly mode: rebuild with the exact same polynomial ordering as _load_inputs_polylift
        - Random/imported mode: rebuild x_all_btf with feature_map + feature_mean/std,
        then run SharedMLP using fixed stored model_state_dict
        - Output: k_jtb shape (J,T,B), J must stay constant
    """
    mode_text = str(cfg_local["k_source_mode"]).strip().lower()

    # Build extended feature tensor (F,T,B).
    feature_map = inputs["feature_map"]
    j_count = int(cfg_local["n_branch"])

    p_ext_ftb = features.build_p_ext_from_p7(
        p_7tb=p_7tb_new,
        feature_map=feature_map,
        scale_io_by_20=bool(cfg_local["scale_io_by_20"]),
    )

    # Convert to model input format (B,T,F), float32 to match theta_N/prior theta_G path.
    x_all_btf = np.transpose(p_ext_ftb, (2, 1, 0)).astype(np.float32)
    # Reuse the fixed feature normalization from the initial data load.
    # feature_mean_f, feature_std_f: shape (F,), where F is x_all_btf.shape[2].
    feature_mean_f = np.asarray(inputs["feature_mean"], dtype=float).reshape(-1)
    feature_std_f = np.asarray(inputs["feature_std"], dtype=float).reshape(-1)

    # Apply normalization to all batches. In imported_nn mode these stats come
    # from theta_N, so regenerated k_jtb stays in the same coordinate system.
    x_all_btf = (
        x_all_btf
        - feature_mean_f.reshape(1, 1, -1).astype(np.float32)
    ) / feature_std_f.reshape(1, 1, -1).astype(np.float32)
    # x_all_btf: shape (B,T,F), normalized regenerated features.
    effective_x_max = inputs.get("x_max", cfg_local.get("x_max", None))
    if effective_x_max is not None:
        x_max = float(effective_x_max)  # scalar clip bound for normalized feature vector x
        x_all_btf = np.clip(x_all_btf, -x_max, x_max)

    # Imported mode path.
    # Random mode path.
    if mode_text == "random_nn" or mode_text == "imported_nn":
        # The following is stored in random_NN
        # inputs["mlp_hidden_dims"] = cfg["hidden_dims"]
        # inputs["mlp_n_branch"] = j_count
        # inputs["mlp_model_state_dict"] = copy.deepcopy(model.state_dict())

        # The following is stored in import_NN
        # inputs["mlp_hidden_dims"] = np.asarray(step1["cfg"]["hidden_dims"],dtype=int)
        # inputs["mlp_n_branch"] = j_count
        # inputs["mlp_model_state_dict"] = np.asarray(step1["model_state_dict"]) # NN parameters 

        # Set deterministic seeds.
        np.random.seed(int(cfg_local["random_seed"]))
        torch.manual_seed(int(cfg_local["random_seed"]))

        # Build untrained MLP model.
        input_dim = int(x_all_btf.shape[2])
        if int(inputs["mlp_n_branch"]) != int(j_count):
            raise ValueError (" HH it is wrong!, j_count = {j_count} but mlp_n_branch = ", int(inputs["mlp_n_branch"]) )
        if tuple(inputs["mlp_hidden_dims"]) != tuple(cfg_local["hidden_dims"]):
            raise ValueError (" HH it is wrong!", tuple(inputs["mlp_hidden_dims"]),  tuple(cfg_local["hidden_dims"]))
        model = SharedMLP(
            input_dim=input_dim,
            n_branch=j_count,
            hidden_dims=tuple(cfg_local["hidden_dims"]),
            hidden_activation=str(inputs.get("mlp_hidden_activation", cfg_local.get("mlp_hidden_activation", "tanh"))),
            output_activation=str(inputs.get("mlp_output_activation", cfg_local.get("mlp_output_activation", "tanh"))),
        )
        if inputs.get("mlp_model_state_dict", None) is None:
            raise ValueError("mlp_model_state_dict is missing in inputs")
        state_dict = copy.deepcopy(inputs["mlp_model_state_dict"])
        model.load_state_dict(state_dict=state_dict, strict=True)

        # Run model once to get fixed k values.
        x_t = torch.tensor(x_all_btf, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            k_btj_t = model(x_t)

        # Convert to numpy and reorder to (J,T,B).
        k_btj = k_btj_t.detach().cpu().numpy().astype(float)
        k_jtb = np.transpose(k_btj, (2, 1, 0)).astype(float)

        return k_jtb
    
    # Poly-lift path. 
    if mode_text == "poly_lifting":
        poly_order = int(cfg_local["poly_order"])
        k_jtb = _build_poly_k_jtb_from_x_all(x_all_btf=x_all_btf,
                                            poly_order=poly_order,
                                            j_count=j_count,
                                            poly_basis_type=str(cfg_local.get("poly_basis_type", "monomial")))
        
        return k_jtb
    # Unsupported mode.
    raise ValueError("k_source_mode must be imported_nn or random_nn or poly_lifting")



def _compute_constraint_diagnostics(
    g_jm: np.ndarray,
    rho_j: np.ndarray,
    rho0_j: np.ndarray,
    eps_j: np.ndarray,
    ms_passivity: int,
) -> dict[str, np.ndarray]:
    """
    Compute decay and passivity diagnostics for solved FIR bank.

    Input:
    - g_jm: np.ndarray, shape (J,M)
    - rho_j: np.ndarray, shape (J,)
    - rho0_j: np.ndarray, shape (J,)
    - eps_j: np.ndarray, shape (J,)
    - ms_passivity: int

    Output keys:
    - decay_envelope: (J,M)
    - decay_margin: (J,M)
    - passivity_response: (J,Ms+1)
    - passivity_min_margin: (J,)
    """
    # Convert inputs.
    g_arr = np.asarray(g_jm, dtype=float)
    rho = np.asarray(rho_j, dtype=float).reshape(-1)
    rho0 = np.asarray(rho0_j, dtype=float).reshape(-1)
    eps = np.asarray(eps_j, dtype=float).reshape(-1)

    # Read dimensions.
    j_count = int(g_arr.shape[0])
    m_count = int(g_arr.shape[1])

    # Build passivity matrix.
    passivity_matrix = step2_min_cvx.build_passivity_matrix(m_count, int(ms_passivity))
    q_count = int(passivity_matrix.shape[0])

    # Allocate outputs.
    decay_envelope = np.zeros((j_count, m_count), dtype=float)
    decay_margin = np.zeros((j_count, m_count), dtype=float)
    passivity_response = np.zeros((j_count, q_count), dtype=float)
    passivity_min_margin = np.zeros(j_count, dtype=float)

    # Compute branch diagnostics.
    for j_index in range(j_count):
        env_m = np.zeros(m_count, dtype=float)
        for m_index in range(m_count):
            env_m[m_index] = rho0[j_index] * (rho[j_index] ** float(m_index))

        decay_envelope[j_index, :] = env_m
        decay_margin[j_index, :] = env_m - np.abs(g_arr[j_index, :])

        resp_q = passivity_matrix @ g_arr[j_index, :]
        passivity_response[j_index, :] = resp_q
        passivity_min_margin[j_index] = float(np.min(resp_q - eps[j_index]))

    # Build output dictionary.
    out: dict[str, np.ndarray] = {}
    out["decay_envelope"] = decay_envelope
    out["decay_margin"] = decay_margin
    out["passivity_response"] = passivity_response
    out["passivity_min_margin"] = passivity_min_margin

    # Return diagnostics dictionary.
    return out


def _mse_on_split(
    y_hat_tb: np.ndarray,
    y_tb: np.ndarray,
    split_idx: np.ndarray,
    zero_cost_first_n: int = 0,
) -> float:
    """
    Compute MSE for one split.

    Input:
    - y_hat_tb: np.ndarray, shape (T,B)
    - y_tb: np.ndarray, shape (T,B)
    - split_idx: np.ndarray, shape (B_split,)

    Output:
    - mse_value: float
    """
    # Normalize split index shape.
    idx = np.asarray(split_idx, dtype=int).reshape(-1)

    # Compute error on selected trajectories.
    n_free = int(zero_cost_first_n)  # scalar int
    t_count = int(np.asarray(y_hat_tb).shape[0])  # scalar T
    if n_free < 0 or n_free >= t_count:
        raise ValueError("zero_cost_first_n must satisfy 0 <= N < T")
    err = np.asarray(y_hat_tb[n_free:, idx] - y_tb[n_free:, idx], dtype=float)

    # Compute mean squared error.
    mse_value = float(np.mean(err * err))

    # Return scalar MSE.
    return mse_value


def _poly_basis_single_x(
    x_f: np.ndarray,
    poly_order: int,
    j_count: int,
    poly_basis_type: str = "monomial",
) -> np.ndarray:
    """
    Build polynomial basis vector for one normalized feature vector.

    Input:
    - x_f: np.ndarray, shape (F,)
    - poly_order: int scalar in {0,1,2,3}
    - j_count: int scalar
    - poly_basis_type: str scalar, "monomial" or "legendre"

    Output:
    - k_j: np.ndarray, shape (J,)
    """
    x_arr = np.asarray(x_f, dtype=float).reshape(-1)  # shape (F,)
    f_count = int(x_arr.shape[0])  # scalar F
    basis_type_text = str(poly_basis_type).strip().lower()
    if basis_type_text not in ("monomial", "legendre"):
        raise ValueError("poly_basis_type must be 'monomial' or 'legendre'")
    out_j = np.zeros((j_count,), dtype=float)  # shape (J,)
    term_idx = 0
    out_j[term_idx] = 1.0
    term_idx += 1

    if basis_type_text == "monomial":
        if poly_order >= 1:
            for i_idx in range(f_count):
                out_j[term_idx] = x_arr[i_idx]
                term_idx += 1

        if poly_order >= 2:
            for i_idx in range(f_count):
                out_j[term_idx] = x_arr[i_idx] * x_arr[i_idx]
                term_idx += 1
            for i_idx in range(f_count):
                for j_idx in range(i_idx + 1, f_count):
                    out_j[term_idx] = x_arr[i_idx] * x_arr[j_idx]
                    term_idx += 1

        if poly_order >= 3:
            for i_idx in range(f_count):
                for j_idx in range(i_idx, f_count):
                    xij = x_arr[i_idx] * x_arr[j_idx]
                    for k_idx in range(j_idx, f_count):
                        out_j[term_idx] = xij * x_arr[k_idx]
                        term_idx += 1

    else:
        l1_f = x_arr
        l2_f = 0.5 * (3.0 * x_arr * x_arr - 1.0)
        l3_f = 0.5 * (5.0 * x_arr * x_arr * x_arr - 3.0 * x_arr)

        if poly_order >= 1:
            for i_idx in range(f_count):
                out_j[term_idx] = l1_f[i_idx]
                term_idx += 1

        if poly_order >= 2:
            for i_idx in range(f_count):
                out_j[term_idx] = l2_f[i_idx]
                term_idx += 1
            for i_idx in range(f_count):
                for j_idx in range(i_idx + 1, f_count):
                    out_j[term_idx] = l1_f[i_idx] * l1_f[j_idx]
                    term_idx += 1

        if poly_order >= 3:
            for i_idx in range(f_count):
                for j_idx in range(i_idx, f_count):
                    for k_idx in range(j_idx, f_count):
                        if i_idx == j_idx and j_idx == k_idx:
                            out_j[term_idx] = l3_f[i_idx]
                        elif i_idx == j_idx:
                            out_j[term_idx] = l2_f[i_idx] * l1_f[k_idx]
                        elif j_idx == k_idx:
                            out_j[term_idx] = l1_f[i_idx] * l2_f[j_idx]
                        else:
                            out_j[term_idx] = l1_f[i_idx] * l1_f[j_idx] * l1_f[k_idx]
                        term_idx += 1

    if term_idx != int(j_count):
        raise ValueError(f"poly basis size mismatch: generated {term_idx}, expected {j_count}")
    return out_j


def _simulate_collect_closed_loop(
    inputs: dict[str, Any],
    g_jm: np.ndarray,
    g_linear_m: np.ndarray,
    cfg_local: dict[str, Any],
) -> dict[str, Any]:
    """
        Closed-loop causal simulation with per-step p_7tb and k_jtb rebuild.

        Input:
        - inputs: dict with u_tb, y_tb, feature metadata and model metadata
        - g_jm: np.ndarray, shape (J,M)
        - g_linear_m: np.ndarray, shape (M,)
        - cfg_local: theta_G config dictionary

        Output:
        - out: dict with the same core keys as _simulate_collect
        plus p_7tb used by closed-loop simulation.
    """
    u_tb = np.asarray(inputs["u_tb"], dtype=float)  # shape (T,B)
    y_tb = np.asarray(inputs["y_tb"], dtype=float)  # shape (T,B)
    feature_map = np.asarray(inputs["feature_map"], dtype=int)  # shape (F,2)
    feature_mean = np.asarray(inputs["feature_mean"], dtype=float).reshape(-1)  # shape (F,)
    feature_std = np.asarray(inputs["feature_std"], dtype=float).reshape(-1)  # shape (F,)
    effective_x_max = inputs.get("x_max", cfg_local.get("x_max", None))  # None or scalar clip bound for normalized x
    split_train_idx = np.asarray(inputs["split_train_idx"], dtype=int).reshape(-1)  # shape (B_train,)
    split_val_idx = np.asarray(inputs["split_val_idx"], dtype=int).reshape(-1)  # shape (B_val,)
    split_test_idx = np.asarray(inputs["split_test_idx"], dtype=int).reshape(-1)  # shape (B_test,)
    g_arr = np.asarray(g_jm, dtype=float)  # shape (J,M)
    g_linear_arr = np.asarray(g_linear_m, dtype=float).reshape(-1)  # shape (M,)

    t_count = int(u_tb.shape[0])  # scalar T
    b_count = int(u_tb.shape[1])  # scalar B
    j_count = int(g_arr.shape[0])  # scalar J
    m_count = int(g_arr.shape[1])  # scalar M
    f_count = int(feature_map.shape[0])  # scalar F
    scale_method_text = str(cfg_local.get("uy_scale_method", "softsign")).strip().lower()  # scalar string
    if scale_method_text not in ("softsign", "divide"):
        raise ValueError("uy_scale_method must be 'softsign' or 'divide'")
    u_cap = float(cfg_local.get("u_max_after_scale", 1.0))  # scalar clip bound for p0
    y_cap = float(cfg_local.get("y_max_after_scale", 1.0))  # scalar clip bound for p1
    if (not np.isfinite(u_cap)) or u_cap <= 0.0:
        raise ValueError("u_max_after_scale must be finite and > 0")
    if (not np.isfinite(y_cap)) or y_cap <= 0.0:
        raise ValueError("y_max_after_scale must be finite and > 0")
    if g_linear_arr.shape[0] != m_count:
        raise ValueError("g_linear_m length must match FIR length M")
    mode_text = str(cfg_local["k_source_mode"]).strip().lower()
    dt_sec = float(cfg_local["iter_p7_ts"])

    # Optional model for imported/random modes.
    model = None
    if mode_text in ("imported_nn", "random_nn"):
        hidden_dims_arr = np.asarray(inputs["mlp_hidden_dims"], dtype=int).reshape(-1) # reshape(-1) = flatten to 1D
        if hidden_dims_arr.size != 2:
            raise ValueError(f"mlp_hidden_dims must have length 2, got {hidden_dims_arr.size}")

        hidden_dims_tuple: tuple[int, int] = (int(hidden_dims_arr[0]), int(hidden_dims_arr[1]))

        model = SharedMLP(
            input_dim=f_count,
            n_branch=j_count,
            hidden_dims=hidden_dims_tuple,
            hidden_activation=str(inputs.get("mlp_hidden_activation", cfg_local.get("mlp_hidden_activation", "tanh"))),
            output_activation=str(inputs.get("mlp_output_activation", cfg_local.get("mlp_output_activation", "tanh"))),
        )
        state_dict = copy.deepcopy(inputs["mlp_model_state_dict"]) # this is loaded in import nn or randmon nn.
        model.load_state_dict(state_dict=state_dict, strict=True)
        model.eval()

    y_hat_tb = np.zeros((t_count, b_count), dtype=float)  # shape (T,B)
    y_nfir_tb = np.zeros((t_count, b_count), dtype=float)  # shape (T,B)
    y_linear_tb = np.zeros((t_count, b_count), dtype=float)  # shape (T,B)
    y_branch_jtb = np.zeros((j_count, t_count, b_count), dtype=float)  # shape (J,T,B)
    k_jtb = np.zeros((j_count, t_count, b_count), dtype=float)  # shape (J,T,B)
    p_7tb = np.zeros((7, t_count, b_count), dtype=float)  # shape (7,T,B)

    for b_index in range(b_count):
        u_t = u_tb[:, b_index]  # shape (T,)
        u_linear_hist_m = np.zeros((m_count,), dtype=float)  # shape (M,)
        if cfg_local["fixed_uy_scale"] == True:
            u_scale = cfg_local["u_scale_fixed"]
        else:
            u_scale = max(1.1 * float(np.max(np.abs(u_t))), 1e-8)  # scalar
        y_scale_run = 1e-8  # scalar running scale for causal normalization
        int_y_prev = 0.0  # scalar integral state at previous step
        nodelay_prev_7 = np.zeros(7, dtype=float)  # shape (7,), previous undelayed channels
        s_hist_jm = np.zeros((j_count, m_count), dtype=float)  # shape (J,M), branch FIR input history

        for t_index in range(t_count):
            # p_curr_7: (7,), causal p at current time step.
            p_curr_7 = np.zeros(7, dtype=float)
            u_now = float(u_t[t_index])  # scalar u(t)
            if m_count > 1:
                u_linear_hist_m[1:] = u_linear_hist_m[0:-1]
            u_linear_hist_m[0] = u_now

            if scale_method_text == "divide":
                p_curr_7[0] = u_now / u_scale
            else:
                p_curr_7[0] = u_now / np.sqrt(u_now * u_now + u_scale * u_scale)
            p_curr_7[0] = float(np.clip(p_curr_7[0], -u_cap, u_cap))
            p_curr_7[1:7] = nodelay_prev_7[1:7]
            p_7tb[:, t_index, b_index] = p_curr_7
        
            # x_feat_f: (F,), feature vector from causal p history with edge-repeat rule. So we now build feature from p(t)
            # Init x_feat_f as zeros, float type 
            x_feat_f = np.zeros(f_count, dtype=float)
            """ Idea
                If feature map structure = fmap =
                [[0,0]
                 [1,0]
                 [1,2]
                 [3,1]]
                 Then it means use dimension 0 and 1 in p(t)
                 then use dimension 1 in p(t) with two step delays 
                 then use dimension 3 in p(t) with 1 step delays

                So, the Feature data at this time t should be
                x_feat_f = [p0(t), p1(t), p1(t-2), p3(t-1)] which is vector of dimension 4. 
                So 
                x_feat_f = [p_7tb_cl[0, t_index, b_index]
                            p_7tb_cl[1, t_index, b_index]
                            p_7tb_cl[1, t_index-2, b_index]
                            p_7tb_cl[3, t_index-1, b_index]]
                If case t_index-1 or t_index-2 < 0,
                we set it to zero, 
            """
            for f_index in range(f_count):
                base_dim = int(feature_map[f_index, 0]) # decide p0 or p1 or p3
                lag = int(feature_map[f_index, 1])  # decide t or t-2 or t-1 
                src_t = t_index - lag # if lag = 2, we need to get t_index-2 in p_7tb_cl
                if src_t < 0:
                    src_t = 0
                x_val = float(p_7tb[base_dim, src_t, b_index]) # one entry of x_feat_f
                if bool(cfg_local["scale_io_by_20"]) and (base_dim == 0 or base_dim == 1):
                    x_val = x_val / 20.0
                x_feat_f[f_index] = x_val

            x_norm_f = (x_feat_f - feature_mean) / feature_std  # shape (F,)
            if effective_x_max is not None:
                x_max = float(effective_x_max)  # scalar clip bound for normalized feature vector x
                x_norm_f = np.clip(x_norm_f, -x_max, x_max)  # shape (F,)

            # Mode-specific k(t).
            if mode_text in ("imported_nn", "random_nn"):
                x_tensor = torch.tensor(
                    x_norm_f.reshape(1, 1, f_count), dtype=torch.float32)  # shape (1,1,F) since model() accepts (B,T,F) inputs. model () output (B,T,J)
                with torch.no_grad():
                    k_11j = model(x_tensor).detach().cpu().numpy() # shape(1,1,J) since model () output (B,T,J), B = T = 1 at this point 
                """ Python trick
                    >>> g = np.zeros((1,1,4))
                    >>> g
                    array([[[0., 0., 0., 0.]]])
                    >>> g.reshape(4)
                    array([0., 0., 0., 0.])
                    >>> g.reshape(4).shape
                    (4,)
                """
                k_now_j = k_11j.reshape(j_count)  # shape (J,)
            elif mode_text == "poly_lifting":
                k_now_j = _poly_basis_single_x(
                    x_f=x_norm_f,
                    poly_order=int(cfg_local["poly_order"]),
                    j_count=j_count,
                    poly_basis_type=str(cfg_local.get("poly_basis_type", "monomial")),
                )  # shape (J,)
            else:
                raise ValueError("Unsupported k_source_mode in closed-loop simulation")

            k_jtb[:, t_index, b_index] = k_now_j

            # s_j: (J,), branch FIR inputs at current time.
            s_j = k_now_j * u_now
            if m_count > 1:
                s_hist_jm[:, 1:] = s_hist_jm[:, 0:-1] # shape history horizontal to discard previous memory
            s_hist_jm[:, 0] = s_j   # s_hist_jm[1,:] = s_1(t) s_1(t-1) .... s_1(t-M+1)

            # g_arr * s_hist_jm is element wise product, sum the along the rows (axis=1) is convotion output at each brance
            v_j = np.sum(g_arr * s_hist_jm, axis=1)  # shape (J,).
            y_branch_j = k_now_j * v_j  # shape (J,)
            y_linear_now = float(np.sum(g_linear_arr * u_linear_hist_m))  # scalar
            y_nfir_now = float(np.sum(y_branch_j))  # scalar
            y_hat_now = y_nfir_now + y_linear_now  # scalar total output

            y_hat_tb[t_index, b_index] = y_hat_now
            y_nfir_tb[t_index, b_index] = y_nfir_now
            y_linear_tb[t_index, b_index] = y_linear_now
            y_branch_jtb[:, t_index, b_index] = y_branch_j

            # Update undelayed channels for next step (delay1 behavior on channels 1..6).
            if cfg_local["fixed_uy_scale"] == True:
                y_scale_run = cfg_local["y_scale_fixed"]
            else:
                y_scale_run = max(y_scale_run, abs(y_hat_now), 1e-8) # decide what is the max scale along the way 
            if scale_method_text == "divide":
                y_scaled_now = y_hat_now / y_scale_run
            else:
                y_scaled_now = y_hat_now / np.sqrt(y_hat_now * y_hat_now + y_scale_run * y_scale_run)
            y_scaled_now = float(np.clip(y_scaled_now, -y_cap, y_cap))
            int_y_now = float(int_y_prev + dt_sec * y_hat_now)

            nodelay_curr_7 = np.zeros(7, dtype=float)  # shape (7,)
            nodelay_curr_7[0] = p_curr_7[0]
            nodelay_curr_7[1] = y_scaled_now
            nodelay_curr_7[2] = np.sin(int_y_now)
            nodelay_curr_7[3] = np.cos(int_y_now)
            nodelay_curr_7[6] = np.tanh(y_hat_now)
            nodelay_prev_7 = nodelay_curr_7
            int_y_prev = int_y_now

    n_free = int(inputs.get("zero_cost_first_n", 0))  # scalar int
    train_mse = _mse_on_split(y_hat_tb, y_tb, split_train_idx, n_free)
    val_mse = _mse_on_split(y_hat_tb, y_tb, split_val_idx, n_free)
    test_mse = _mse_on_split(y_hat_tb, y_tb, split_test_idx, n_free)

    out: dict[str, Any] = {}
    out["y_hat_tb"] = y_hat_tb
    out["y_nfir_tb"] = y_nfir_tb
    out["y_linear_tb"] = y_linear_tb
    out["y_branch_jtb"] = y_branch_jtb
    out["k_jtb"] = k_jtb
    out["p_7tb"] = p_7tb
    out["y_pre_train_batch"] = y_hat_tb[:, split_train_idx]
    out["y_train_batch"] = y_tb[:, split_train_idx]
    out["y_pre_val_batch"] = y_hat_tb[:, split_val_idx]
    out["y_val_batch"] = y_tb[:, split_val_idx]
    out["y_pre_test_batch"] = y_hat_tb[:, split_test_idx]
    out["y_test_batch"] = y_tb[:, split_test_idx]
    out["y_pre_train_branch_batch"] = y_branch_jtb[:, :, split_train_idx]
    out["y_pre_val_branch_batch"] = y_branch_jtb[:, :, split_val_idx]
    out["y_pre_test_branch_batch"] = y_branch_jtb[:, :, split_test_idx]
    out["k_pre_train_branch_batch"] = k_jtb[:, :, split_train_idx]
    out["k_pre_val_branch_batch"] = k_jtb[:, :, split_val_idx]
    out["k_pre_test_branch_batch"] = k_jtb[:, :, split_test_idx]
    out["train_mse"] = float(train_mse)
    out["val_mse"] = float(val_mse)
    out["test_mse"] = float(test_mse)
    return out


def _simulate_collect(inputs: dict[str, Any], g_jm: np.ndarray, g_linear_m: np.ndarray) -> dict[str, Any]:
    """
        Simulate full dataset and collect split-local arrays.

        Input:
        - inputs: normalized input dictionary
        - g_jm: np.ndarray, shape (J,M)

        Output dictionary keys:
        - y_hat_tb, y_branch_jtb, s_jtb, v_jtb, g_oim
        - split-local y/y_pred arrays
        - split-local branch arrays
        - split-local k arrays
        - train_mse, val_mse, test_mse
    """
    # Read inputs.
    u_tb = np.asarray(inputs["u_tb"], dtype=float)
    y_tb = np.asarray(inputs["y_tb"], dtype=float)
    k_jtb = np.asarray(inputs["k_jtb"], dtype=float)

    split_train_idx = np.asarray(inputs["split_train_idx"], dtype=int).reshape(-1)
    split_val_idx = np.asarray(inputs["split_val_idx"], dtype=int).reshape(-1)
    split_test_idx = np.asarray(inputs["split_test_idx"], dtype=int).reshape(-1)

    # Run forward simulation.
    # print('In  _simulate_collect. Start calling nfir_forward_diagonal_mimo')
    # print(' ')
    # timer1 = time.time()
    # y_hat_tb, y_branch_jtb, s_jtb, v_jtb, g_oim = step2_min_mimo.nfir_forward_diagonal_mimo(
    #     k_jtb=k_jtb,
    #     u_tb=u_tb,
    #     g_jm=np.asarray(g_jm, dtype=float),
    # )
    y_nfir_tb, y_branch_jtb, s_jtb, v_jtb, g_oim = step2_min_mimo.nfir_forward_diagonal_mimo(
    k_jtb=k_jtb,
    u_tb=u_tb,
    g_jm=np.asarray(g_jm, dtype=float),  # shape (J,M)
    )
    y_linear_tb = step2_min_mimo.linear_fir_forward(
        u_tb=u_tb,
        g_linear_m=np.asarray(g_linear_m, dtype=float),  # shape (M,)
    )
    y_hat_tb = y_nfir_tb + y_linear_tb  # shape (T,B), total output
    # print('In  _simulate_collect. Finish calling nfir_forward_diagonal_mimo. Time = ' ,time.time()- timer1)
    # print(' ')

    # Compute split MSE values.
    n_free = int(inputs.get("zero_cost_first_n", 0))  # scalar int
    train_mse = _mse_on_split(y_hat_tb, y_tb, split_train_idx, n_free)
    val_mse = _mse_on_split(y_hat_tb, y_tb, split_val_idx, n_free)
    test_mse = _mse_on_split(y_hat_tb, y_tb, split_test_idx, n_free)

    # Build output dictionary.
    out: dict[str, Any] = {}

    out["y_hat_tb"] = y_hat_tb          # shape (T,B), total output
    out["y_nfir_tb"] = y_nfir_tb        # shape (T,B)
    out["y_linear_tb"] = y_linear_tb    # shape (T,B)
    out["y_branch_jtb"] = y_branch_jtb  # shape (J,T,B)

    out["s_jtb"] = s_jtb
    out["v_jtb"] = v_jtb
    out["g_oim"] = g_oim

    out["y_pre_train_batch"] = y_hat_tb[:, split_train_idx]
    out["y_train_batch"] = y_tb[:, split_train_idx]
    out["y_pre_val_batch"] = y_hat_tb[:, split_val_idx]
    out["y_val_batch"] = y_tb[:, split_val_idx]
    out["y_pre_test_batch"] = y_hat_tb[:, split_test_idx]
    out["y_test_batch"] = y_tb[:, split_test_idx]

    out["y_pre_train_branch_batch"] = y_branch_jtb[:, :, split_train_idx]
    out["y_pre_val_branch_batch"] = y_branch_jtb[:, :, split_val_idx]
    out["y_pre_test_branch_batch"] = y_branch_jtb[:, :, split_test_idx]

    out["k_pre_train_branch_batch"] = k_jtb[:, :, split_train_idx]
    out["k_pre_val_branch_batch"] = k_jtb[:, :, split_val_idx]
    out["k_pre_test_branch_batch"] = k_jtb[:, :, split_test_idx]

    out["train_mse"] = float(train_mse)
    out["val_mse"] = float(val_mse)
    out["test_mse"] = float(test_mse)

    # Return simulation output dictionary.
    return out


def _to_mat_safe_obj(value: Any) -> Any:
    """
    Convert Python object to MATLAB savemat-safe object recursively.

    Rule:
    - None -> empty string ""
    - dict/list/tuple -> recurse
    - others -> return as-is
    """
    # Convert None to empty string.
    if value is None:
        return ""

    # Recurse for dictionary.
    if isinstance(value, dict):
        out_dict: dict[str, Any] = {}
        for key_name, key_value in value.items():
            key_text = str(key_name)
            if len(key_text) > 0:
                if not key_text[0].isalpha():
                    key_text = "k_" + key_text
            out_dict[key_text] = _to_mat_safe_obj(key_value)
        return out_dict

    # Recurse for list/tuple.
    if isinstance(value, (list, tuple)):
        out_list: list[Any] = []
        for item in value:
            out_list.append(_to_mat_safe_obj(item))
        return out_list

    # Base case.
    return value


def _save_outputs_min(result: dict[str, Any], out_dir: str | Path, run_name: str) -> tuple[Path, Path]:
    """
    Save theta_G result to PKL and MATLAB files.

    Input:
    - result: result dictionary
    - out_dir: output directory
    - run_name: output base name

    Output:
    - (pkl_path, mat_path)
    """
    # Resolve output directory path and create it.
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Build output file paths.
    pkl_path = out_dir_path / f"{run_name}.pkl"
    mat_path = out_dir_path / f"{run_name}_train.mat"

    # Save full result dictionary to pickle.
    with pkl_path.open("wb") as file_obj:
        pickle.dump(result, file_obj)

    # Build compact MATLAB structure.
    data_mat: dict[str, Any] = {}

    # Copy scalar/text metadata.
    data_mat["schema_version"] = str(result["schema_version"])
    data_mat["mode"] = str(result["mode"])
    data_mat["run_name"] = str(result["run_name"])
    data_mat["k_source_mode"] = str(result["k_source_mode"])
    
    data_mat["g_linear_m"] = np.asarray(result["g_linear_m"], dtype=float)
    data_mat["y_nfir_tb"] = np.asarray(result["y_nfir_tb"], dtype=float)
    data_mat["y_linear_tb"] = np.asarray(result["y_linear_tb"], dtype=float)
    if "y_nfir_tb_cl" in result:
        data_mat["y_nfir_tb_cl"] = np.asarray(result["y_nfir_tb_cl"], dtype=float)
    if "y_linear_tb_cl" in result:
        data_mat["y_linear_tb_cl"] = np.asarray(result["y_linear_tb_cl"], dtype=float)

    # Copy core tensors/arrays.
    data_mat["u_tb"] = np.asarray(result["u_tb"], dtype=float)
    data_mat["y_tb"] = np.asarray(result["y_tb"], dtype=float)
    data_mat["p_7tb"] = np.asarray(result["p_7tb"], dtype=float)
    data_mat["k_jtb"] = np.asarray(result["k_jtb"], dtype=float)
    if "p_7tb_cl" in result:
        data_mat["p_7tb_cl"] = np.asarray(result["p_7tb_cl"], dtype=float)
    if "k_jtb_cl" in result:
        data_mat["k_jtb_cl"] = np.asarray(result["k_jtb_cl"], dtype=float)

    data_mat["g_bank"] = np.asarray(result["g_bank"], dtype=float)
    data_mat["y_hat_tb"] = np.asarray(result["y_hat_tb"], dtype=float)
    data_mat["y_branch_jtb"] = np.asarray(result["y_branch_jtb"], dtype=float)
    if "y_hat_tb_cl" in result:
        data_mat["y_hat_tb_cl"] = np.asarray(result["y_hat_tb_cl"], dtype=float)
    if "y_branch_jtb_cl" in result:
        data_mat["y_branch_jtb_cl"] = np.asarray(result["y_branch_jtb_cl"], dtype=float)

    data_mat["split_train_idx"] = np.asarray(result["split_train_idx"], dtype=np.int32)
    data_mat["split_val_idx"] = np.asarray(result["split_val_idx"], dtype=np.int32)
    data_mat["split_test_idx"] = np.asarray(result["split_test_idx"], dtype=np.int32)

    data_mat["feature_map"] = np.asarray(result["feature_map"], dtype=int)
    data_mat["feature_mean"] = np.asarray(result["feature_mean"], dtype=float)
    data_mat["feature_std"] = np.asarray(result["feature_std"], dtype=float)
    data_mat["x_max"] = np.array([[np.nan if result.get("x_max", None) is None else float(result["x_max"])]], dtype=float)

    data_mat["y_pre_train_batch"] = np.asarray(result["y_pre_train_batch"], dtype=float)
    data_mat["y_train_batch"] = np.asarray(result["y_train_batch"], dtype=float)
    data_mat["y_pre_val_batch"] = np.asarray(result["y_pre_val_batch"], dtype=float)
    data_mat["y_val_batch"] = np.asarray(result["y_val_batch"], dtype=float)
    data_mat["y_pre_test_batch"] = np.asarray(result["y_pre_test_batch"], dtype=float)
    data_mat["y_test_batch"] = np.asarray(result["y_test_batch"], dtype=float)
    if "y_pre_train_batch_cl" in result:
        data_mat["y_pre_train_batch_cl"] = np.asarray(result["y_pre_train_batch_cl"], dtype=float)
    if "y_pre_val_batch_cl" in result:
        data_mat["y_pre_val_batch_cl"] = np.asarray(result["y_pre_val_batch_cl"], dtype=float)
    if "y_pre_test_batch_cl" in result:
        data_mat["y_pre_test_batch_cl"] = np.asarray(result["y_pre_test_batch_cl"], dtype=float)

    data_mat["y_pre_train_branch_batch"] = np.asarray(result["y_pre_train_branch_batch"], dtype=float)
    data_mat["y_pre_val_branch_batch"] = np.asarray(result["y_pre_val_branch_batch"], dtype=float)
    data_mat["y_pre_test_branch_batch"] = np.asarray(result["y_pre_test_branch_batch"], dtype=float)
    if "y_pre_train_branch_batch_cl" in result:
        data_mat["y_pre_train_branch_batch_cl"] = np.asarray(result["y_pre_train_branch_batch_cl"], dtype=float)
    if "y_pre_val_branch_batch_cl" in result:
        data_mat["y_pre_val_branch_batch_cl"] = np.asarray(result["y_pre_val_branch_batch_cl"], dtype=float)
    if "y_pre_test_branch_batch_cl" in result:
        data_mat["y_pre_test_branch_batch_cl"] = np.asarray(result["y_pre_test_branch_batch_cl"], dtype=float)

    if "k_pre_train_branch_batch" in result:
        data_mat["k_pre_train_branch_batch"] = np.asarray(result["k_pre_train_branch_batch"], dtype=float)
    if "k_pre_val_branch_batch" in result:
        data_mat["k_pre_val_branch_batch"] = np.asarray(result["k_pre_val_branch_batch"], dtype=float)
    if "k_pre_test_branch_batch" in result:
        data_mat["k_pre_test_branch_batch"] = np.asarray(result["k_pre_test_branch_batch"], dtype=float)
    if "k_pre_train_branch_batch_cl" in result:
        data_mat["k_pre_train_branch_batch_cl"] = np.asarray(result["k_pre_train_branch_batch_cl"], dtype=float)
    if "k_pre_val_branch_batch_cl" in result:
        data_mat["k_pre_val_branch_batch_cl"] = np.asarray(result["k_pre_val_branch_batch_cl"], dtype=float)
    if "k_pre_test_branch_batch_cl" in result:
        data_mat["k_pre_test_branch_batch_cl"] = np.asarray(result["k_pre_test_branch_batch_cl"], dtype=float)

    data_mat["t_compile"] = np.array([[float(result["t_compile"])]], dtype=float)
    data_mat["t_solve"] = np.array([[float(result["t_solve"])]], dtype=float)

    data_mat["train_mse"] = np.array([[float(result["train_mse"])]], dtype=float)
    data_mat["val_mse"] = np.array([[float(result["val_mse"])]], dtype=float)
    data_mat["test_mse"] = np.array([[float(result["test_mse"])]], dtype=float)
    if "train_mse_cl" in result:
        data_mat["train_mse_cl"] = np.array([[float(result["train_mse_cl"])]], dtype=float)
    if "val_mse_cl" in result:
        data_mat["val_mse_cl"] = np.array([[float(result["val_mse_cl"])]], dtype=float)
    if "test_mse_cl" in result:
        data_mat["test_mse_cl"] = np.array([[float(result["test_mse_cl"])]], dtype=float)
    # Iteration-history exports (compact arrays for easy MATLAB plotting).
    if "iter_train_mse_hist" in result:
        data_mat["iter_train_mse_hist"] = np.asarray(result["iter_train_mse_hist"], dtype=float)
    if "iter_val_mse_hist" in result:
        data_mat["iter_val_mse_hist"] = np.asarray(result["iter_val_mse_hist"], dtype=float)
    if "iter_test_mse_hist" in result:
        data_mat["iter_test_mse_hist"] = np.asarray(result["iter_test_mse_hist"], dtype=float)
    if "iter_train_mse_cl_hist" in result:
        data_mat["iter_train_mse_cl_hist"] = np.asarray(result["iter_train_mse_cl_hist"], dtype=float)
    if "iter_val_mse_cl_hist" in result:
        data_mat["iter_val_mse_cl_hist"] = np.asarray(result["iter_val_mse_cl_hist"], dtype=float)
    if "iter_test_mse_cl_hist" in result:
        data_mat["iter_test_mse_cl_hist"] = np.asarray(result["iter_test_mse_cl_hist"], dtype=float)
    if "iter_y_hat_tb_hist" in result:
        data_mat["iter_y_hat_tb_hist"] = np.asarray(result["iter_y_hat_tb_hist"], dtype=float)
    if "iter_y_hat_tb_cl_hist" in result:
        data_mat["iter_y_hat_tb_cl_hist"] = np.asarray(result["iter_y_hat_tb_cl_hist"], dtype=float)
    if "iter_k_jtb_hist" in result:
        data_mat["iter_k_jtb_hist"] = np.asarray(result["iter_k_jtb_hist"], dtype=float)
    if "iter_k_jtb_cl_hist" in result:
        data_mat["iter_k_jtb_cl_hist"] = np.asarray(result["iter_k_jtb_cl_hist"], dtype=float)
    if "iter_p_7tb_hist" in result:
        data_mat["iter_p_7tb_hist"] = np.asarray(result["iter_p_7tb_hist"], dtype=float)
    if "iter_p_7tb_cl_hist" in result:
        data_mat["iter_p_7tb_cl_hist"] = np.asarray(result["iter_p_7tb_cl_hist"], dtype=float)
    if "iter_y_pre_train_batch_hist" in result:
        data_mat["iter_y_pre_train_batch_hist"] = np.asarray(result["iter_y_pre_train_batch_hist"], dtype=float)
    if "iter_y_pre_val_batch_hist" in result:
        data_mat["iter_y_pre_val_batch_hist"] = np.asarray(result["iter_y_pre_val_batch_hist"], dtype=float)
    if "iter_y_pre_test_batch_hist" in result:
        data_mat["iter_y_pre_test_batch_hist"] = np.asarray(result["iter_y_pre_test_batch_hist"], dtype=float)
    if "iter_y_pre_train_batch_cl_hist" in result:
        data_mat["iter_y_pre_train_batch_cl_hist"] = np.asarray(result["iter_y_pre_train_batch_cl_hist"], dtype=float)
    if "iter_y_pre_val_batch_cl_hist" in result:
        data_mat["iter_y_pre_val_batch_cl_hist"] = np.asarray(result["iter_y_pre_val_batch_cl_hist"], dtype=float)
    if "iter_y_pre_test_batch_cl_hist" in result:
        data_mat["iter_y_pre_test_batch_cl_hist"] = np.asarray(result["iter_y_pre_test_batch_cl_hist"], dtype=float)
    if "iter_g_bank_hist" in result:
        data_mat["iter_g_bank_hist"] = np.asarray(result["iter_g_bank_hist"], dtype=float)
    if "iter_history" in result:
        data_mat["iter_history"] = _to_mat_safe_obj(result["iter_history"])
    if "iter_history" in result:
        data_mat["iter_count"] = np.array([[int(len(result["iter_history"]))]], dtype=np.int32)

    data_mat["solver_status"] = str(result["solver_status"])
    data_mat["opt_value"] = np.array([[float(result["opt_value"])]], dtype=float)

    data_mat["decay_envelope"] = np.asarray(result["decay_envelope"], dtype=float)
    data_mat["decay_margin"] = np.asarray(result["decay_margin"], dtype=float)
    data_mat["passivity_response"] = np.asarray(result["passivity_response"], dtype=float)
    data_mat["passivity_min_margin"] = np.asarray(result["passivity_min_margin"], dtype=float)

    data_mat["cfg"] = _to_mat_safe_obj(dict(result["cfg"]))

    # Save with top-level struct key '<run_name>_train'.
    top_key = f"{run_name}_train"
    scipy.io.savemat(str(mat_path), {top_key: data_mat})

    # Return output file paths.
    return pkl_path, mat_path


def run_step2_min(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """
        Run theta_G optimization end-to-end.

        Input:
        - cfg: dict[str, Any] | None
        If None, defaults for imported_nn mode are used.

        Output:
        - result: dict[str, Any]
        Core keys include:
        u_tb, y_tb, k_jtb, g_bank, y_hat_tb, y_branch_jtb,
        split indices, split-local prediction arrays,
        train/val/test MSE, decay/passivity diagnostics,
        solver_status, opt_value, cfg, pkl_path, mat_path.
    """
    # Use default config when caller passes None.
    if cfg is None:
        cfg_local = build_default_config_min("imported_nn")
    else:
        cfg_local = dict(cfg)
        cfg_local.setdefault("enable_parallel_fir", True)
        cfg_local.setdefault("zero_cost_first_n", 0)

    # Validate config dictionary.
    _validate_cfg(cfg_local)

    # Get what mode is using 
    mode_text = str(cfg_local["k_source_mode"]).strip().lower()
    """ Python trick 
        Given a dictionary e.g. cfg_local = {"n_branch":5}
        print( cfg_local.get("n_branch", None) ) = 5
        print( cfg_local.get("hh", None) ) = None -- so return none is no such key

        Hence: cfg_local.get("n_branch", None) is None  -- make sure there is no such key
    """
    if mode_text == "imported_nn":
        if cfg_local.get("n_branch", None) is None or cfg_local.get("m_fir", None) is None:  
            # This is the case I use imported nn but forgot to specify n_branch etc. 
            j_count_ref, m_fir_ref = _read_step1_dims(str(cfg_local["source_step1_pkl"]))
            cfg_local["n_branch"] = int(j_count_ref)
            cfg_local["m_fir"] = int(m_fir_ref)

    # The above code got J and M from step 1 pkl
    # Now put local varaibles on it. 
    j_count = int(cfg_local["n_branch"])
    m_fir = int(cfg_local["m_fir"])

    # Build decay rate and eps
    rho_j = _to_branch_vector(cfg_local.get("rho_j", None), j_count, 
                              float(cfg_local.get("rho_default",0.93)))
    rho0_j = _to_branch_vector(cfg_local.get("rho0_j", None), j_count, float(cfg_local.get("rho0_default", 100.0)))
    eps_j = _to_branch_vector(cfg_local.get("eps_j", None), j_count, float(cfg_local.get("eps_default", 5e-3)))

    # Load normalized inputs.
    inputs, j_loaded, m_loaded = _load_inputs(cfg_local)

    # Validate loaded J/M match resolved config J/M.
    if int(j_loaded) != int(j_count):
        raise ValueError(f"Loaded J={j_loaded} does not match cfg J={j_count}")
    if int(m_loaded) != int(m_fir):
        raise ValueError(f"Loaded M={m_loaded} does not match cfg M={m_fir}")

    # Prepare arrays for CVX solve and optional diagnostics print.
    u_tb_arr = np.asarray(inputs["u_tb"], dtype=float)
    y_tb_arr = np.asarray(inputs["y_tb"], dtype=float)
    k_jtb_arr = np.asarray(inputs["k_jtb"], dtype=float)
    train_idx_arr = np.asarray(inputs["split_train_idx"], dtype=int)
    rho_j_arr = np.asarray(rho_j, dtype=float)
    rho0_j_arr = np.asarray(rho0_j, dtype=float)
    eps_j_arr = np.asarray(eps_j, dtype=float)
    zero_cost_first_n = int(cfg_local["zero_cost_first_n"])  # scalar int
    t_count_loaded = int(u_tb_arr.shape[0])  # scalar T
    if zero_cost_first_n < 0 or zero_cost_first_n >= t_count_loaded:
        raise ValueError("zero_cost_first_n must satisfy 0 <= N < T")
    inputs["zero_cost_first_n"] = zero_cost_first_n

    if bool(cfg_local["verbose_solver"]):
        t_count = int(u_tb_arr.shape[0])
        b_count = int(u_tb_arr.shape[1]) # current is 20 
        b_train = int(train_idx_arr.shape[0]) # current is 16
        # MILESTONE 1 TODO:
        # Extend pre-cvx print with iteration settings:
        # - n_refine_iter
        # - n_total_iter = n_refine_iter + 1
        # - iter_p7_ts
        print(
            f"[step2_min][pre-cvx] mode={mode_text} J={j_count} M={m_fir} "
            f"T={t_count} B={b_count} B_train={b_train} "
            f"is_passive={bool(cfg_local['is_passive'])} ms_passivity={int(cfg_local['ms_passivity'])} "
            f"n_refine_iter={cfg_local['n_refine_iter']} n_total_iter={int(cfg_local['n_refine_iter'])+1} iter_p7_ts={float(cfg_local['iter_p7_ts'])}"
        )
        print(
            f"[step2_min][pre-cvx] shapes: u_tb={u_tb_arr.shape} y_tb={y_tb_arr.shape} "
            f"k_jtb={k_jtb_arr.shape} train_idx={train_idx_arr.shape} "
            f"rho_j={rho_j_arr.shape} rho0_j={rho0_j_arr.shape} eps_j={eps_j_arr.shape}"
        )

    # MILESTONE 4 TODO:
    # Replace the single-pass block below with iterative refinement loop:
    # for iter_idx in range(0, n_refine_iter + 1):
    #   1) solve_step2_cvx_min with current k_jtb, i.e. solve the theta_G subproblem.
    #   2) _simulate_collect with current k_jtb and solved g_jm
    #   3) _compute_constraint_diagnostics
    #   4) append one iter_history item
    #   5) if not last iter: rebuild p_7tb from (u_tb, y_hat_tb), then rebuild k_jtb
    # Keep u_tb and y_tb fixed across all iterations.
    # Keep final top-level outputs mapped to the last iteration.

    # Solve FIR coefficients using CVX.
    
    iter_history: list[dict[str, Any]] = []

    iter_idx_max = int(cfg_local["n_refine_iter"])
    y_hat_tb_pre_ite = np.zeros_like(y_tb_arr) # (T,B) B is total number of batches 20, rather than just training batches 16
    p_7tb_new = np.asarray(inputs["p_7tb"], dtype=float)
    sim_out_last: dict[str, Any] | None = None
    sim_out_cl_last: dict[str, Any] | None = None

    for iter_idx in range(0, iter_idx_max + 1):
        print('Start cvx.')
        # print(' ')
        timer1 = time.time()
        if iter_idx > 0: # means iter_idx_max > 0, so we are in iterative mode 
            iter_yhat_prev_weight = float(cfg_local["iter_yhat_prev_weight"])
            y_for_p7_tb = (
                iter_yhat_prev_weight * y_hat_tb_pre_ite
                + (1.0 - iter_yhat_prev_weight) * y_tb_arr
            )  # shape (T,B), convex blend for repeated CVX p_7tb rebuild
            p_7tb_new = _p_7tb_regen(u_tb=u_tb_arr, 
                                     y_hat_tb= y_for_p7_tb, 
                                     ts=float(cfg_local["iter_p7_ts"]),
                                     add_noise=bool(cfg_local["iter_yhat_add_noise"]),
                                     noise_snr_db=float(cfg_local["iter_yhat_noise_snr_db"]),
                                     noise_seed=cfg_local["iter_yhat_noise_seed"],
                                     lpf_enable=bool(cfg_local["iter_yhat_lpf_enable"]),
                                     lpf_tau=float(cfg_local["iter_yhat_lpf_tau"]),
                                     cfg_local=cfg_local)
            k_jtb_arr = _rebuild_k_jtb_all_mode(inputs=inputs,cfg_local=cfg_local,p_7tb_new=p_7tb_new)

        solve_out = step2_min_cvx.solve_step2_cvx_min(
            u_tb=u_tb_arr,
            y_tb=y_tb_arr,
            k_jtb=k_jtb_arr,
            train_idx=train_idx_arr,
            m_fir=int(m_fir),
            l2reg=float(cfg_local["l2reg"]),
            is_passive=bool(cfg_local["is_passive"]),
            rho_j=rho_j_arr,
            rho0_j=rho0_j_arr,
            eps_j=eps_j_arr,
            ms_passivity=int(cfg_local["ms_passivity"]),
            solver_name=str(cfg_local["solver_name"]),
            verbose_solver=bool(cfg_local["verbose_solver"]),
            enable_parallel_fir=bool(cfg_local["enable_parallel_fir"]),
            zero_cost_first_n=zero_cost_first_n,
        )
        

        print('Solve finish. Takes ', time.time() - timer1, ' seconds.')
        # print(' ')
        # Read solved FIR bank.
        g_jm = np.asarray(solve_out["g_jm"], dtype=float)
        g_linear_m = np.asarray(solve_out["g_linear_m"], dtype=float)  # shape (M,)

        # print('Start simulating all data and collect split arrays.')
        # print(' ')
        # timer1 = time.time()
        # Simulate all data and collect split arrays.
        inputs_iter = dict(inputs)
        inputs_iter["k_jtb"] = k_jtb_arr
        inputs_iter["p_7tb"] = p_7tb_new
        # sim_out = _simulate_collect(inputs=inputs_iter, g_jm=g_jm)
        sim_out = _simulate_collect(inputs=inputs_iter, g_jm=g_jm, g_linear_m=g_linear_m)
        sim_out_cl = _simulate_collect_closed_loop(
            inputs=inputs_iter,
            g_jm=np.asarray(g_jm, dtype=float),
            g_linear_m=np.asarray(g_linear_m, dtype=float),
            cfg_local=cfg_local,
        )
        # print('Simulate all data done. Takes ', time.time() - timer1, ' seconds.')
        # print(' ')

        y_hat_tb_pre_ite = np.asarray(sim_out["y_hat_tb"], dtype=float)
        sim_out_last = sim_out
        sim_out_cl_last = sim_out_cl

        # Compute diagnostics from solved FIR.
        diag_out = _compute_constraint_diagnostics(
            g_jm=g_jm,
            rho_j=rho_j,
            rho0_j=rho0_j,
            eps_j=eps_j,
            ms_passivity=int(cfg_local["ms_passivity"]),
        )

        iter_item = {
        "iter_idx": int(iter_idx),
        "solver_status": str(solve_out["solver_status"]),
        "opt_value": float(solve_out["opt_value"]),
        "train_mse": float(sim_out["train_mse"]),
        "val_mse": float(sim_out["val_mse"]),
        "test_mse": float(sim_out["test_mse"]),
        "train_mse_cl": float(sim_out_cl["train_mse"]),
        "val_mse_cl": float(sim_out_cl["val_mse"]),
        "test_mse_cl": float(sim_out_cl["test_mse"]),
        "t_compile": float(solve_out["t_compile"]),
        "t_solve": float(solve_out["t_solve"]),
        "g_bank": np.asarray(g_jm, dtype=float).copy(),      # (J,M)
        "y_hat_tb": np.asarray(sim_out["y_hat_tb"], dtype=float).copy(),  # (T,B)
        "y_hat_tb_cl": np.asarray(sim_out_cl["y_hat_tb"], dtype=float).copy(),  # (T,B)
        "y_branch_jtb_cl": np.asarray(sim_out_cl["y_branch_jtb"], dtype=float).copy(),  # (J,T,B)
        "k_jtb": np.asarray(k_jtb_arr, dtype=float).copy(), # (J,T,B)
        # "k_jtb_cl": np.asarray(sim_out_cl["k_jtb"], dtype=float).copy(), # (J,T,B)
        "p_7tb": np.asarray(p_7tb_new, dtype=float).copy(), # (7,T,B)
        "p_7tb_cl": np.asarray(sim_out_cl["p_7tb"], dtype=float).copy(), # (7,T,B)
        "y_pre_train_batch_cl": np.asarray(sim_out_cl["y_pre_train_batch"], dtype=float).copy(),
        "y_pre_val_batch_cl": np.asarray(sim_out_cl["y_pre_val_batch"], dtype=float).copy(),
        "y_pre_test_batch_cl": np.asarray(sim_out_cl["y_pre_test_batch"], dtype=float).copy(),
        "y_pre_train_branch_batch_cl": np.asarray(sim_out_cl["y_pre_train_branch_batch"], dtype=float).copy(),
        "y_pre_val_branch_batch_cl": np.asarray(sim_out_cl["y_pre_val_branch_batch"], dtype=float).copy(),
        "y_pre_test_branch_batch_cl": np.asarray(sim_out_cl["y_pre_test_branch_batch"], dtype=float).copy(),
        "k_pre_train_branch_batch_cl": np.asarray(sim_out_cl["k_pre_train_branch_batch"], dtype=float).copy(),
        "k_pre_val_branch_batch_cl": np.asarray(sim_out_cl["k_pre_val_branch_batch"], dtype=float).copy(),
        "k_pre_test_branch_batch_cl": np.asarray(sim_out_cl["k_pre_test_branch_batch"], dtype=float).copy(),
        }
        iter_history.append(iter_item)

    if sim_out_last is None:
        raise ValueError("sim_out_last is None; iteration loop produced no outputs")
    if sim_out_cl_last is None:
        raise ValueError("sim_out_cl_last is None; iteration loop produced no closed-loop outputs")
    # Build compact result dictionary.
    

    result: dict[str, Any] = {}
    result["iter_history"] = iter_history
    result["iter_train_mse_hist"] = np.asarray([it["train_mse"] for it in iter_history], dtype=float)   # (Niter,)
    result["iter_val_mse_hist"]   = np.asarray([it["val_mse"] for it in iter_history], dtype=float)
    result["iter_test_mse_hist"]  = np.asarray([it["test_mse"] for it in iter_history], dtype=float)
    result["iter_train_mse_cl_hist"] = np.asarray([it["train_mse_cl"] for it in iter_history], dtype=float)
    result["iter_val_mse_cl_hist"]   = np.asarray([it["val_mse_cl"] for it in iter_history], dtype=float)
    result["iter_test_mse_cl_hist"]  = np.asarray([it["test_mse_cl"] for it in iter_history], dtype=float)
    # Iteration raw tensors for direct MATLAB post-processing (e.g., RMSE boxplots).
    # Dimensions:
    # - iter_y_hat_tb_hist: (T,B,Niter)
    # - iter_y_hat_tb_cl_hist: (T,B,Niter)
    # - iter_k_jtb_hist / iter_k_jtb_cl_hist: (J,T,B,Niter)
    # - iter_p_7tb_hist / iter_p_7tb_cl_hist: (7,T,B,Niter)
    # - iter_y_pre_*_batch*_hist: (T,B_split,Niter)
    result["iter_y_hat_tb_hist"] = np.stack([it["y_hat_tb"] for it in iter_history], axis=2)
    result["iter_y_hat_tb_cl_hist"] = np.stack([it["y_hat_tb_cl"] for it in iter_history], axis=2)
    result["iter_k_jtb_hist"] = np.stack([it["k_jtb"] for it in iter_history], axis=3)
    # result["iter_k_jtb_cl_hist"] = np.stack([it["k_jtb_cl"] for it in iter_history], axis=3)
    result["iter_p_7tb_hist"] = np.stack([it["p_7tb"] for it in iter_history], axis=3)
    result["iter_p_7tb_cl_hist"] = np.stack([it["p_7tb_cl"] for it in iter_history], axis=3)
    result["iter_y_pre_train_batch_hist"] = np.stack([it["y_hat_tb"][:, train_idx_arr] for it in iter_history], axis=2)
    result["iter_y_pre_val_batch_hist"] = np.stack([it["y_hat_tb"][:, np.asarray(inputs["split_val_idx"], dtype=int)] for it in iter_history], axis=2)
    result["iter_y_pre_test_batch_hist"] = np.stack([it["y_hat_tb"][:, np.asarray(inputs["split_test_idx"], dtype=int)] for it in iter_history], axis=2)
    result["iter_y_pre_train_batch_cl_hist"] = np.stack([it["y_pre_train_batch_cl"] for it in iter_history], axis=2)
    result["iter_y_pre_val_batch_cl_hist"] = np.stack([it["y_pre_val_batch_cl"] for it in iter_history], axis=2)
    result["iter_y_pre_test_batch_cl_hist"] = np.stack([it["y_pre_test_batch_cl"] for it in iter_history], axis=2)
    result["iter_g_bank_hist"]    = np.stack([it["g_bank"] for it in iter_history], axis=2)             # (J,M,Niter)


    result["schema_version"] = str(cfg_local.get("schema_version", "nfir8e_step2_min_v1"))
    result["mode"] = str(cfg_local.get("mode", "step2_fir"))
    result["run_name"] = str(cfg_local["run_name"])

    result["k_source_mode"] = str(inputs["k_source_mode"])
    result["source_step1_path"] = inputs["source_step1_path"]
    result["source_data_mat_path"] = inputs["source_data_mat_path"]

    result["cfg"] = dict(cfg_local)
    result["zero_cost_first_n"] = int(zero_cost_first_n)
    # MILESTONE 5 TODO:
    # Add result["iter_history"] with one dict per iteration:
    # - Scalars: iter_idx, solver_status, opt_value, train_mse, val_mse, test_mse, t_compile, t_solve
    # - Key tensors: g_bank, y_hat_tb, p_7tb, k_jtb
    # Keep MAT export stable with final-iteration top-level fields unchanged.

    result["u_tb"] = np.asarray(inputs["u_tb"], dtype=float)
    result["y_tb"] = np.asarray(inputs["y_tb"], dtype=float)
    result["p_7tb"] = np.asarray(p_7tb_new, dtype=float)
    result["p_7tb_cl"] = np.asarray(sim_out_cl_last["p_7tb"], dtype=float)
    result["k_jtb"] = np.asarray(k_jtb_arr, dtype=float)
    # result["k_jtb_cl"] = np.asarray(sim_out_cl_last["k_jtb"], dtype=float)


    result["feature_map"] = np.asarray(inputs["feature_map"], dtype=int)
    result["feature_mean"] = np.asarray(inputs["feature_mean"], dtype=float)
    result["feature_std"] = np.asarray(inputs["feature_std"], dtype=float)
    result["x_max"] = inputs.get("x_max", cfg_local.get("x_max", None))
    # Keep NN weights available in the theta_G result for analytic tests.
    # Shape detail:
    # - model_state_dict is a torch state-dict mapping parameter names to tensors.
    # - For poly_lifting mode this stays None.
    if inputs.get("mlp_model_state_dict", None) is None:
        result["model_state_dict"] = None
    else:
        result["model_state_dict"] = copy.deepcopy(inputs["mlp_model_state_dict"])

    result["split_train_idx"] = np.asarray(inputs["split_train_idx"], dtype=int)
    result["split_val_idx"] = np.asarray(inputs["split_val_idx"], dtype=int)
    result["split_test_idx"] = np.asarray(inputs["split_test_idx"], dtype=int)

    result["g_bank"] = np.asarray(g_jm, dtype=float)
    result["g_linear_m"] = np.asarray(g_linear_m, dtype=float)    # shape (M,)

    result["solver_status"] = str(solve_out["solver_status"])
    result["opt_value"] = float(solve_out["opt_value"])

    result["t_compile"] = float(solve_out["t_compile"])
    result["t_solve"] = float(solve_out["t_solve"])
    

    result["y_hat_tb"] = np.asarray(sim_out_last["y_hat_tb"], dtype=float)
    result["y_nfir_tb"] = np.asarray(sim_out_last["y_nfir_tb"], dtype=float)      # shape (T,B)
    result["y_linear_tb"] = np.asarray(sim_out_last["y_linear_tb"], dtype=float)  # shape (T,B)

    result["y_hat_tb_cl"] = np.asarray(sim_out_cl_last["y_hat_tb"], dtype=float)
    result["y_nfir_tb_cl"] = np.asarray(sim_out_cl_last["y_nfir_tb"], dtype=float)
    result["y_linear_tb_cl"] = np.asarray(sim_out_cl_last["y_linear_tb"], dtype=float)
    result["y_branch_jtb"] = np.asarray(sim_out_last["y_branch_jtb"], dtype=float)
    result["y_branch_jtb_cl"] = np.asarray(sim_out_cl_last["y_branch_jtb"], dtype=float)

    result["y_pre_train_batch"] = np.asarray(sim_out_last["y_pre_train_batch"], dtype=float)
    result["y_train_batch"] = np.asarray(sim_out_last["y_train_batch"], dtype=float)
    result["y_pre_val_batch"] = np.asarray(sim_out_last["y_pre_val_batch"], dtype=float)
    result["y_val_batch"] = np.asarray(sim_out_last["y_val_batch"], dtype=float)
    result["y_pre_test_batch"] = np.asarray(sim_out_last["y_pre_test_batch"], dtype=float)
    result["y_test_batch"] = np.asarray(sim_out_last["y_test_batch"], dtype=float)
    result["y_pre_train_batch_cl"] = np.asarray(sim_out_cl_last["y_pre_train_batch"], dtype=float)
    result["y_pre_val_batch_cl"] = np.asarray(sim_out_cl_last["y_pre_val_batch"], dtype=float)
    result["y_pre_test_batch_cl"] = np.asarray(sim_out_cl_last["y_pre_test_batch"], dtype=float)

    result["y_pre_train_branch_batch"] = np.asarray(sim_out_last["y_pre_train_branch_batch"], dtype=float)
    result["y_pre_val_branch_batch"] = np.asarray(sim_out_last["y_pre_val_branch_batch"], dtype=float)
    result["y_pre_test_branch_batch"] = np.asarray(sim_out_last["y_pre_test_branch_batch"], dtype=float)
    result["y_pre_train_branch_batch_cl"] = np.asarray(sim_out_cl_last["y_pre_train_branch_batch"], dtype=float)
    result["y_pre_val_branch_batch_cl"] = np.asarray(sim_out_cl_last["y_pre_val_branch_batch"], dtype=float)
    result["y_pre_test_branch_batch_cl"] = np.asarray(sim_out_cl_last["y_pre_test_branch_batch"], dtype=float)

    # result["k_pre_train_branch_batch"] = np.asarray(sim_out_last["k_pre_train_branch_batch"], dtype=float)
    # result["k_pre_val_branch_batch"] = np.asarray(sim_out_last["k_pre_val_branch_batch"], dtype=float)
    # result["k_pre_test_branch_batch"] = np.asarray(sim_out_last["k_pre_test_branch_batch"], dtype=float)
    # result["k_pre_train_branch_batch_cl"] = np.asarray(sim_out_cl_last["k_pre_train_branch_batch"], dtype=float)
    # result["k_pre_val_branch_batch_cl"] = np.asarray(sim_out_cl_last["k_pre_val_branch_batch"], dtype=float)
    # result["k_pre_test_branch_batch_cl"] = np.asarray(sim_out_cl_last["k_pre_test_branch_batch"], dtype=float)

    result["train_mse"] = float(sim_out_last["train_mse"])
    result["val_mse"] = float(sim_out_last["val_mse"])
    result["test_mse"] = float(sim_out_last["test_mse"])
    result["train_mse_cl"] = float(sim_out_cl_last["train_mse"])
    result["val_mse_cl"] = float(sim_out_cl_last["val_mse"])
    result["test_mse_cl"] = float(sim_out_cl_last["test_mse"])

    result["decay_envelope"] = np.asarray(diag_out["decay_envelope"], dtype=float)
    result["decay_margin"] = np.asarray(diag_out["decay_margin"], dtype=float)
    result["passivity_response"] = np.asarray(diag_out["passivity_response"], dtype=float)
    result["passivity_min_margin"] = np.asarray(diag_out["passivity_min_margin"], dtype=float)

    result["rho_j"] = np.asarray(rho_j, dtype=float)
    result["rho0_j"] = np.asarray(rho0_j, dtype=float)
    result["eps_j"] = np.asarray(eps_j, dtype=float)
    result["ms_passivity"] = int(cfg_local["ms_passivity"])
    result["l2reg"] = float(cfg_local["l2reg"])

    if cfg_local["save_full_diagnostics"] == False: # whether save all details 
        result["y_branch_jtb"] = np.zeros(1)
        result["y_branch_jtb_cl"] = np.zeros(1)

        result["y_pre_train_branch_batch"] = np.zeros(1)
        result["y_pre_val_branch_batch"] = np.zeros(1)
        result["y_pre_test_branch_batch"] = np.zeros(1)
        result["y_pre_train_branch_batch_cl"] = np.zeros(1)
        result["y_pre_val_branch_batch_cl"] = np.zeros(1)
        result["y_pre_test_branch_batch_cl"] = np.zeros(1)

        result["iter_history"] = []
        result["iter_train_mse_hist"] = np.zeros(1)
        result["iter_val_mse_hist"]   = np.zeros(1)
        result["iter_test_mse_hist"]  = np.zeros(1)
        result["iter_train_mse_cl_hist"] = np.zeros(1)
        result["iter_val_mse_cl_hist"]   = np.zeros(1)
        result["iter_test_mse_cl_hist"]  = np.zeros(1)

        result["iter_y_hat_tb_hist"] = np.zeros(1)
        result["iter_y_hat_tb_cl_hist"] = np.zeros(1)
        result["iter_k_jtb_hist"] = np.zeros(1)
        result["iter_k_jtb_cl_hist"] =np.zeros(1)
        result["iter_p_7tb_hist"] = np.zeros(1)
        result["iter_p_7tb_cl_hist"] = np.zeros(1)
        result["iter_y_pre_train_batch_hist"] = np.zeros(1)
        result["iter_y_pre_val_batch_hist"] = np.zeros(1)
        result["iter_y_pre_test_batch_hist"] = np.zeros(1)
        result["iter_y_pre_train_batch_cl_hist"] =np.zeros(1)
        result["iter_y_pre_val_batch_cl_hist"] = np.zeros(1)
        result["iter_y_pre_test_batch_cl_hist"] = np.zeros(1)
        result["iter_g_bank_hist"]    = np.zeros(1)           # (J,M,Niter)

        result["k_jtb"] =  np.zeros(1)  
        result["k_jtb_cl"] =  np.zeros(1)  
        # result["p_7tb"] = np.zeros(1)  
        # result["p_7tb_cl"] = np.zeros(1)  




    # Save outputs.
    pkl_path, mat_path = _save_outputs_min(
        result=result,
        out_dir=str(cfg_local["out_dir"]),
        run_name=str(cfg_local["run_name"]),
    )

    # Attach output paths.
    result["pkl_path"] = str(pkl_path)
    result["mat_path"] = str(mat_path)

    # Return full result dictionary.
    return result
