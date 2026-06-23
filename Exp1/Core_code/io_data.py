"""MATLAB/Python data I/O helpers for Exp1 NFIR runs.

Role in the workflow:
- Load MATLAB training data into the common tensor convention used by theta_G
  and theta_N: u_tb and y_tb have shape (T,B), and p_7tb has shape (7,T,B).
- Rebuild scheduling channels from input/output data when closed-loop rollout
  needs model-generated outputs.
- Save theta_N outputs to both pickle and MATLAB ``*_train.mat`` files so
  later theta_G/theta_N runs and MATLAB analysis scripts can consume the same
  results.

Open-source compatibility note:
- Field names in saved MATLAB structs are intentionally preserved. Comments may
  call these theta_N outputs, but runtime field names are not changed.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import pickle

import numpy as np
import scipy.io

import torch


def utc_now_iso() -> str:
    """
    Return current UTC time string in fixed format.

    Input:
    - none

    Output:
    - time_text: str
      Example format: "2026-04-06T14:30:12Z"

    Dimension notes:
    - scalar string only.
    """
    # Get current UTC datetime object.
    now_utc = datetime.now(timezone.utc)
    # Convert to fixed text format used in exports.
    time_text = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    # Return the formatted text.
    return time_text


def load_training_mat(mat_path: str | Path) -> dict:
    """
    Load MATLAB training data and return arrays with fixed dimensions.

    Input:
    - mat_path: str or Path
      Path to MATLAB file that contains key "dta".

    Output dictionary keys and dimensions:
    - "u_tb": numpy.ndarray, shape (T, B), dtype float64
      T = number of time samples, B = number of trajectories (batches).
    - "y_tb": numpy.ndarray, shape (T, B), dtype float64
    - "p_7tb": numpy.ndarray, shape (7, T, B), dtype float64
      First axis has exactly 7 scheduling channels.
    - "n_time": int, equals T
    - "n_batch": int, equals B
    - "source_mat_path": str

    Dimension conventions used in this project:
    - t axis: time index, length T
    - b axis: trajectory index, length B
    - 7 axis: base scheduling dimension index in [0..6]
    """

    mat_path = Path(mat_path) # Ensure mat_path is a Path object.

    # Load matlab data
    mat_data = scipy.io.loadmat(str(mat_path))

    # check whether dta attribute is there
    if "dta" not in mat_data:
        raise ValueError((f" Mat file miss attribute dta: {mat_path}"))

    dta = mat_data["dta"]

    # Read scalar dimensions from MATLAB struct fields.
    n_time = int(dta["N_train"][0, 0][0][0])
    n_batch = int(dta["n_train"][0, 0][0][0])

    # Read raw arrays from MATLAB struct fields.
    ipt_train_mat = dta["ipt_train_mat"][0, 0]
    opt_train_mat = dta["opt_train_mat"][0, 0]
    p_train_delay_mat = dta["p_train_delay_mat"][0, 0]

    # Slice and cast to get the exact arrays used by the legacy theta_N path.
    # u_tb has shape (T, B).
    u_tb = np.asarray(ipt_train_mat[0, 0:n_time, 0:n_batch], dtype=float)
    # y_tb has shape (T, B).
    y_tb = np.asarray(opt_train_mat[0, 0:n_time, 0:n_batch], dtype=float)
    # p_7tb has shape (7, T, B).
    p_7tb = np.asarray(p_train_delay_mat[:, 0:n_time, 0:n_batch], dtype=float)

    # Validate loaded dimensions strictly.
    if u_tb.shape != (n_time, n_batch):
        raise ValueError(
            f"u_tb shape mismatch: expected {(n_time, n_batch)}, got {u_tb.shape}"
        )
    if y_tb.shape != (n_time, n_batch):
        raise ValueError(
            f"y_tb shape mismatch: expected {(n_time, n_batch)}, got {y_tb.shape}"
        )
    if p_7tb.shape != (7, n_time, n_batch):
        raise ValueError(
            f"p_7tb shape mismatch: expected {(7, n_time, n_batch)}, got {p_7tb.shape}"
        )

    # Build output dictionary.
    out = {}
    out["u_tb"] = u_tb
    out["y_tb"] = y_tb
    out["p_7tb"] = p_7tb
    out["n_time"] = n_time
    out["n_batch"] = n_batch
    out["source_mat_path"] = str(mat_path)

    return out


def build_p7_from_u_y(
    u_tb: np.ndarray,
    y_tb: np.ndarray,
    ts: float,
    fixed_uy_scale: bool,
    u_scale_fixed: float,
    y_scale_fixed: float,
    uy_scale_method: str,
    u_max_after_scale: float,
    y_max_after_scale: float,
) -> np.ndarray:
    """
    Build delayed scheduling tensor p_7tb from u_tb and y_tb.

    This function is tested in tests/test_build_p7_from_u_y.py. Test passed.

    Input dimensions:
    - u_tb: np.ndarray, shape (T, B)
    - y_tb: np.ndarray, shape (T, B)
    - ts: float scalar, sample time
    - fixed_uy_scale: bool scalar, True->fixed scales, False->dynamic global scales
    - u_scale_fixed: float scalar
    - y_scale_fixed: float scalar
    - uy_scale_method: str scalar, "softsign" or "divide"
    - u_max_after_scale: float scalar, clip bound for scaled u channel
    - y_max_after_scale: float scalar, clip bound for scaled y channel

    Output:
    - p_7tb_delay: np.ndarray, shape (7, T, B), dtype float64

    Channel layout before delay:
    - ch0: u_scaled
    - ch1: y_scaled
    - ch2: sin(int_y)
    - ch3: cos(int_y)
    - ch4: 0
    - ch5: 0
    - ch6: tanh(y)
    Then apply delay1 on channels 1..6 only.
    """
    # u_arr, y_arr: (T, B)
    u_arr = np.asarray(u_tb, dtype=float)
    y_arr = np.asarray(y_tb, dtype=float)
    if u_arr.ndim != 2 or y_arr.ndim != 2:
        raise ValueError("u_tb and y_tb must both have shape (T,B)")
    if u_arr.shape != y_arr.shape:
        raise ValueError("u_tb and y_tb must have the same shape (T,B)")

    # Validate scalar options.
    ts_val = float(ts)
    if ts_val <= 0.0:
        raise ValueError("ts must be > 0")
    scale_method_text = str(uy_scale_method).strip().lower()
    if scale_method_text not in ("softsign", "divide"):
        raise ValueError("uy_scale_method must be 'softsign' or 'divide'")
    u_cap = float(u_max_after_scale)
    y_cap = float(y_max_after_scale)
    if (not np.isfinite(u_cap)) or u_cap <= 0.0:
        raise ValueError("u_max_after_scale must be finite and > 0")
    if (not np.isfinite(y_cap)) or y_cap <= 0.0:
        raise ValueError("y_max_after_scale must be finite and > 0")

    # Resolve scale values used for p0 and p1.
    if bool(fixed_uy_scale):
        u_scale = float(u_scale_fixed)  # scalar
        y_scale = float(y_scale_fixed)  # scalar
    else:
        u_scale = max(1.1 * float(np.max(np.abs(u_arr))), 1e-8)  # scalar
        y_scale = max(1.1 * float(np.max(np.abs(y_arr))), 1e-8)  # scalar

    # Build scaled channels with selected method.
    if scale_method_text == "divide":
        u_scaled_tb = u_arr / u_scale  # (T, B)
        y_scaled_tb = y_arr / y_scale  # (T, B)
    else:
        u_scaled_tb = u_arr / np.sqrt(u_arr * u_arr + u_scale * u_scale)  # (T, B)
        y_scaled_tb = y_arr / np.sqrt(y_arr * y_arr + y_scale * y_scale)  # (T, B)

    # Apply post-scale clipping bounds.
    u_scaled_tb = np.clip(u_scaled_tb, -u_cap, u_cap)  # (T, B)
    y_scaled_tb = np.clip(y_scaled_tb, -y_cap, y_cap)  # (T, B)

    # int_y_tb: (T, B), integral of y over time.
    int_y_tb = ts_val * np.cumsum(y_arr, axis=0)

    # Build undelayed p channels.
    t_count = int(u_arr.shape[0])  # scalar T
    b_count = int(u_arr.shape[1])  # scalar B
    p_7tb_nodelay = np.zeros((7, t_count, b_count), dtype=float)  # (7, T, B)
    p_7tb_nodelay[0, :, :] = u_scaled_tb
    p_7tb_nodelay[1, :, :] = y_scaled_tb
    p_7tb_nodelay[2, :, :] = np.sin(int_y_tb)
    p_7tb_nodelay[3, :, :] = np.cos(int_y_tb)
    p_7tb_nodelay[6, :, :] = np.tanh(y_arr)

    # Apply delay1 to channels 1..6 only, keep channel 0 unchanged.
    p_7tb_delay = p_7tb_nodelay.copy()  # (7, T, B)
    for ch_idx in range(1, 7):
        p_7tb_delay[ch_idx, 0, :] = 0.0
        p_7tb_delay[ch_idx, 1:, :] = p_7tb_nodelay[ch_idx, 0:-1, :]

    return p_7tb_delay


def save_step1_outputs(result: dict, out_dir: str | Path, run_name: str) -> tuple[Path, Path]:
    """
    Save theta_N result to both pickle and MATLAB formats.

    Input:
    - result: dict
      Must include key arrays and metadata from training.
      Required dimensions inside `result`:
      - u_tb: (T, B)
      - y_tb: (T, B)
      - p_7tb: (7, T, B)
      - y_hat_tb: (T, B)
      - y_branch_jtb: (J, T, B)
      - k_jtb: (J, T, B)
      - g_bank: (J, M)
      - split_train_idx: (n_tr,)
      - split_val_idx: (n_va,)
      - split_test_idx: (n_te,)
    - out_dir: output folder path
    - run_name: output base name

    Output:
    - (pkl_path, mat_path): tuple[Path, Path]

    Notes:
    - MATLAB file contains one top-level struct named "{run_name}_train".
    - Field names follow old theta_N export naming for compatibility.
    """

    save_full_diagnostics = bool(result["cfg"].get("save_full_diagnostics", True))
    empty_diag = np.zeros(1, dtype=float)

    # Convert and create output directory.
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build output file paths.
    pkl_path = out_dir / f"{run_name}.pkl"
    mat_path = out_dir / f"{run_name}_train.mat"

    # Move model state tensors to CPU explicitly.
    model_state_cpu = {}
    for state_key in result["model_state_dict"]:
        # Read one tensor/value from state dict.
        state_value = result["model_state_dict"][state_key]
        # If tensor, move/detach to CPU for stable pickling.
        if torch.is_tensor(state_value):
            model_state_cpu[state_key] = state_value.detach().cpu()
        else:
            model_state_cpu[state_key] = state_value

    # Copy result dictionary so we do not mutate the caller's object.
    pkl_data = dict(result)
    # Replace state dict with CPU-safe version.
    pkl_data["model_state_dict"] = model_state_cpu

    # Write pickle file.
    with pkl_path.open("wb") as f:
        pickle.dump(pkl_data, f)

    # Read split indices as integer arrays.
    tr = np.asarray(result["split_train_idx"], dtype=int)
    va = np.asarray(result["split_val_idx"], dtype=int)
    te = np.asarray(result["split_test_idx"], dtype=int)

    # Build MATLAB struct fields g_0, g_1, ..., g_{J-1}.
    g_bank = np.asarray(result["g_bank"], dtype=float)
    g_mat = {}
    j_count = int(g_bank.shape[0])
    for j in range(j_count):
        key_name = f"g_{j}"
        g_mat[key_name] = g_bank[j, :]

    # Build MATLAB export structure with old-compatible field names.
    data_mat = {}
    data_mat["schema_version"] = str(result["schema_version"])
    data_mat["mode"] = str(result["mode"])
    data_mat["timestamp_utc"] = str(result["timestamp_utc"])

    # Train subset arrays.
    data_mat["y_pre_train_batch"] = np.asarray(result["y_hat_tb"])[:, tr]
    data_mat["y_train_batch"] = np.asarray(result["y_tb"])[:, tr]
    if save_full_diagnostics:
        data_mat["y_pre_train_branch_batch"] = np.asarray(result["y_branch_jtb"])[:, :, tr]
    else:
        data_mat["y_pre_train_branch_batch"] = empty_diag
    data_mat["k_pre_train_branch_batch"] = np.asarray(result["k_jtb"])[:, :, tr]
    data_mat["u_train_batch"] = np.asarray(result["u_tb"])[:, tr]
    data_mat["int_u_train_batch"] = np.zeros_like(np.asarray(result["u_tb"])[:, tr])
    data_mat["p_train_delay_batch"] = np.asarray(result["p_7tb"])[:, :, tr]

    # Validation subset arrays.
    data_mat["y_pre_val_batch"] = np.asarray(result["y_hat_tb"])[:, va]
    data_mat["y_val_batch"] = np.asarray(result["y_tb"])[:, va]

    # Test subset arrays.
    data_mat["y_pre_test_batch"] = np.asarray(result["y_hat_tb"])[:, te]
    data_mat["y_test_batch"] = np.asarray(result["y_tb"])[:, te]

    # Optional closed-loop fields.
    if "y_hat_tb_cl" in result:
        data_mat["y_pre_train_batch_cl"] = np.asarray(result["y_hat_tb_cl"])[:, tr]
        data_mat["y_pre_val_batch_cl"] = np.asarray(result["y_hat_tb_cl"])[:, va]
        data_mat["y_pre_test_batch_cl"] = np.asarray(result["y_hat_tb_cl"])[:, te]
    if "y_branch_jtb_cl" in result:
        if save_full_diagnostics:
            data_mat["y_pre_train_branch_batch_cl"] = np.asarray(result["y_branch_jtb_cl"])[:, :, tr]
            data_mat["y_pre_val_branch_batch_cl"] = np.asarray(result["y_branch_jtb_cl"])[:, :, va]
            data_mat["y_pre_test_branch_batch_cl"] = np.asarray(result["y_branch_jtb_cl"])[:, :, te]
        else:
            data_mat["y_pre_train_branch_batch_cl"] = empty_diag
            data_mat["y_pre_val_branch_batch_cl"] = empty_diag
            data_mat["y_pre_test_branch_batch_cl"] = empty_diag

    if "k_jtb_cl" in result:
        data_mat["k_pre_train_branch_batch_cl"] = np.asarray(result["k_jtb_cl"])[:, :, tr]
        data_mat["k_pre_val_branch_batch_cl"] = np.asarray(result["k_jtb_cl"])[:, :, va]
        data_mat["k_pre_test_branch_batch_cl"] = np.asarray(result["k_jtb_cl"])[:, :, te]
    if "p_7tb_cl" in result:
        data_mat["p_train_delay_batch_cl"] = np.asarray(result["p_7tb_cl"])[:, :, tr]
    if "train_mse_cl" in result:
        data_mat["train_mse_cl"] = np.array([[float(result["train_mse_cl"])]], dtype=float)
        data_mat["val_mse_cl"] = np.array([[float(result["val_mse_cl"])]], dtype=float)
        data_mat["test_mse_cl"] = np.array([[float(result["test_mse_cl"])]], dtype=float)

    # History arrays.
    data_mat["nn_train_loss_history"] = np.asarray(result["history_train_loss"])
    data_mat["nn_val_loss_history"] = np.asarray(result["history_val_loss"])
    data_mat["nn_test_loss_history"] = np.asarray(result["history_test_loss"])
    data_mat["nn_epoch_history"] = np.asarray(result["history_epoch"])
    data_mat["nn_open_loop_train_time_sec"] = np.array([[float(result["nn_open_loop_train_time_sec"])]], dtype=float)
    data_mat["nn_bptt_train_time_sec"] = np.array([[float(result["nn_bptt_train_time_sec"])]], dtype=float)
    data_mat["nn_train_time_sec"] = np.array([[float(result["nn_train_time_sec"])]], dtype=float)


    # Optional BPTT history arrays.
    if "bptt_history_epoch" in result:
        data_mat["bptt_history_epoch"] = np.asarray(result["bptt_history_epoch"], dtype=int)
        data_mat["bptt_history_train_loss"] = np.asarray(result["bptt_history_train_loss"], dtype=float)
        data_mat["bptt_history_val_loss"] = np.asarray(result["bptt_history_val_loss"], dtype=float)
        data_mat["bptt_history_test_loss"] = np.asarray(result["bptt_history_test_loss"], dtype=float)
        data_mat["bptt_best_epoch"] = np.array([[int(result["bptt_best_epoch"])]], dtype=int)
        data_mat["bptt_best_val_loss"] = np.array([[float(result["bptt_best_val_loss"])]], dtype=float)

    # Best metrics stored as 2D MATLAB scalars.
    data_mat["best_epoch"] = np.array([[int(result["best_epoch"])]], dtype=int)
    data_mat["best_val_loss"] = np.array([[float(result["best_val_loss"])]], dtype=float)

    # Feature and split metadata.
    data_mat["feature_map"] = np.asarray(result["feature_map"], dtype=int)
    data_mat["feature_mean"] = np.asarray(result["feature_mean"], dtype=float)
    data_mat["feature_std"] = np.asarray(result["feature_std"], dtype=float)
    data_mat["x_max"] = np.array([[np.nan if result.get("x_max", None) is None else float(result["x_max"])]], dtype=float)
    data_mat["split_train_idx"] = tr.astype(np.int32)
    data_mat["split_val_idx"] = va.astype(np.int32)
    data_mat["split_test_idx"] = te.astype(np.int32)
    data_mat["split_train_idx_1based"] = tr.astype(np.int32) + 1
    data_mat["split_val_idx_1based"] = va.astype(np.int32) + 1
    data_mat["split_test_idx_1based"] = te.astype(np.int32) + 1

    # FIR metadata.
    data_mat["g_NFIR_mat"] = g_mat
    data_mat["g_bank"] = np.asarray(result["g_bank"], dtype=float)
    data_mat["fir_taus"] = np.asarray(result["fir_taus"], dtype=float)
    data_mat["fir_gains"] = np.asarray(result["fir_gains"], dtype=float)
    data_mat["fir_source_type"] = str(result["fir_source_type"])
    data_mat["fir_source_path"] = str(result["fir_source_path"])

    # Lineage metadata (fixed to "None" for minimal path).
    data_mat["lineage_prev_step1_path"] = str(result["lineage_prev_step1_path"])
    data_mat["lineage_prev_step2_path"] = str(result["lineage_prev_step2_path"])
    data_mat["lineage_freeze_split_path"] = str(result["lineage_freeze_split_path"])
    if "skip_open_loop_training" in result:
        data_mat["skip_open_loop_training"] = np.array([[bool(result["skip_open_loop_training"])]], dtype=bool)
    if "used_initial_model_state_dict" in result:
        data_mat["used_initial_model_state_dict"] = np.array([[bool(result["used_initial_model_state_dict"])]], dtype=bool)

    # Configuration dictionary. MATLAB cannot save Python None, so use an empty string
    # inside the MAT copy only; the PKL above keeps the original Python None.
    cfg_mat = dict(result["cfg"])
    if cfg_mat.get("x_max", None) is None:
        cfg_mat["x_max"] = ""
    data_mat["cfg"] = cfg_mat

    # Save MATLAB file with top-level struct name run_name + "_train".
    savemat_dict = {}
    top_key = f"{run_name}_train"
    savemat_dict[top_key] = data_mat
    scipy.io.savemat(str(mat_path), savemat_dict)

    # Return paths for caller summary.
    return pkl_path, mat_path
