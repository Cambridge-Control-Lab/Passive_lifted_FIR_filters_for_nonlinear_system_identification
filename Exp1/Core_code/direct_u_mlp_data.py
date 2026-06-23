"""Data helpers for the Exp1 direct-u MLP baseline.

Role in the workflow:
- This baseline maps delayed input samples directly to output samples and does
  not implement the passive lifted FIR/NFIR structure of arXiv:2508.05279v2.
- It is kept as comparison code for Exp1, so comments here focus on data
  layout, splitting, normalization, and saved-output compatibility.

Notation:
- T: number of time samples
- B: number of trajectories
- D: number of delayed input features, equal to u_delay_steps + 1
- N: number of flattened samples, typically selected_batches * T
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pickle

import numpy as np
import scipy.io

from . import features
from . import io_data


def build_default_data_config() -> dict[str, Any]:
    """
    Build default config shared by direct u -> y MLP baselines.

    Dimension notes:
    - u_tb: shape (T,B)
    - x_btd: shape (B,T,D), D = u_delay_steps + 1
    - X_2d: shape (N,D), N = number of selected batches * T
    - y_1d: shape (N,)
    """
    cfg: dict[str, Any] = {}

    cfg["run_name"] = "run_direct_u_mlp"
    cfg["schema_version"] = "direct_u_mlp_v1"
    cfg["mode"] = "direct_u_mlp"

    # Input memory: d=0 means the model input is only u(t).
    cfg["u_delay_steps"] = 0

    # Scale u before building delay-window features.
    cfg["u_scale_fixed"] = 22.0
    cfg["uy_scale_method"] = "divide"  # "divide" or "softsign"
    cfg["u_max_after_scale"] = 100000.0

    # Normalize delayed input features after feature construction.
    cfg["feature_norm_mode"] = "zscore"  # "none" or "zscore"
    cfg["x_max"] = None  # None or float > 0, clips normalized x to [-x_max,x_max]

    # Normalize y only inside training. Saved predictions are in original y units.
    cfg["target_norm_mode"] = "zscore"  # "none" or "zscore"

    # Batch split follows the theta_N convention used by the NFIR training path.
    cfg["train_val_test_split"] = (15, 3, 2)
    cfg["split_seed"] = 42
    cfg["shuffle_split"] = False

    cfg["random_seed"] = 1
    return cfg


def utc_now_iso() -> str:
    """Return timestamp string used in saved outputs."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_data_config(cfg: dict[str, Any]) -> None:
    """Validate only important direct-MLP config entries."""
    if int(cfg["u_delay_steps"]) < 0:
        raise ValueError("u_delay_steps must be >= 0")
    if float(cfg["u_scale_fixed"]) <= 0.0:
        raise ValueError("u_scale_fixed must be > 0")
    if str(cfg["uy_scale_method"]).lower() not in ("divide", "softsign"):
        raise ValueError("uy_scale_method must be 'divide' or 'softsign'")
    if float(cfg["u_max_after_scale"]) <= 0.0:
        raise ValueError("u_max_after_scale must be > 0")
    if str(cfg["feature_norm_mode"]).lower() not in ("none", "zscore"):
        raise ValueError("feature_norm_mode must be 'none' or 'zscore'")
    if str(cfg["target_norm_mode"]).lower() not in ("none", "zscore"):
        raise ValueError("target_norm_mode must be 'none' or 'zscore'")
    if cfg.get("x_max", None) is not None:
        x_max = float(cfg["x_max"])
        if (not np.isfinite(x_max)) or x_max <= 0.0:
            raise ValueError("x_max must be None or a positive finite scalar")


def _norm_stats_2d(x_2d: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute column normalization stats.

    Input:
    - x_2d: shape (N,D)
    - mode: "none" or "zscore"

    Output:
    - mean_d: shape (D,)
    - std_d: shape (D,)
    """
    x_arr = np.asarray(x_2d, dtype=float)
    d_count = int(x_arr.shape[1])
    mode_text = str(mode).lower()
    if mode_text == "none":
        return np.zeros(d_count, dtype=float), np.ones(d_count, dtype=float)
    if mode_text == "zscore":
        mean_d = np.mean(x_arr, axis=0)
        std_d = np.std(x_arr, axis=0)
        for d_idx in range(d_count):
            if std_d[d_idx] < 1e-8:
                std_d[d_idx] = 1.0
        return mean_d.astype(float), std_d.astype(float)
    raise ValueError(f"Unsupported normalization mode: {mode}")


def scale_u_for_direct_mlp(u_tb: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """
    Scale input u before delayed feature construction.

    Input:
    - u_tb: shape (T,B)

    Output:
    - u_scaled_tb: shape (T,B)
    """
    u_arr = np.asarray(u_tb, dtype=float)  # (T,B)
    u_scale = float(cfg["u_scale_fixed"])
    method_text = str(cfg["uy_scale_method"]).lower()
    if method_text == "divide":
        u_scaled_tb = u_arr / u_scale
    elif method_text == "softsign":
        u_scaled_tb = u_arr / np.sqrt(u_arr * u_arr + u_scale * u_scale)
    else:
        raise ValueError("uy_scale_method must be 'divide' or 'softsign'")

    u_cap = float(cfg["u_max_after_scale"])
    u_scaled_tb = np.clip(u_scaled_tb, -u_cap, u_cap)  # (T,B)
    return u_scaled_tb


def build_u_delay_features(u_tb: np.ndarray, u_delay_steps: int) -> np.ndarray:
    """
    Build direct MLP features from delayed u only.

    Input:
    - u_tb: shape (T,B)
    - u_delay_steps: scalar d >= 0

    Output:
    - x_btd: shape (B,T,D), D=d+1
      x_btd[b,t,lag] = u(t-lag,b), with edge-hold at t=0.
    """
    u_arr = np.asarray(u_tb, dtype=float)  # (T,B)
    if u_arr.ndim != 2:
        raise ValueError("u_tb must have shape (T,B)")

    t_count = int(u_arr.shape[0])
    b_count = int(u_arr.shape[1])
    d_count = int(u_delay_steps) + 1
    x_btd = np.zeros((b_count, t_count, d_count), dtype=float)  # (B,T,D)

    for b_idx in range(b_count):
        for t_idx in range(t_count):
            for lag_idx in range(d_count):
                src_t = t_idx - lag_idx
                if src_t < 0:
                    src_t = 0
                x_btd[b_idx, t_idx, lag_idx] = u_arr[src_t, b_idx]

    return x_btd


def split_direct_mlp_data(u_tb: np.ndarray, y_tb: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Build train/val/test arrays for direct u -> y MLP.

    Input:
    - u_tb: shape (T,B)
    - y_tb: shape (T,B)

    Output keys and dimensions:
    - x_all_btd: shape (B,T,D), normalized features
    - y_all_bt: shape (B,T), original target
    - X_train: shape (B_train*T,D)
    - y_train: shape (B_train*T,), original target
    - train_idx/val_idx/test_idx: batch index vectors
    - x_mean_d/x_std_d: shape (D,)
    - y_mean/y_std: scalars for optional target zscore
    """
    _validate_data_config(cfg)

    u_arr = np.asarray(u_tb, dtype=float)  # (T,B)
    y_arr = np.asarray(y_tb, dtype=float)  # (T,B)
    if u_arr.ndim != 2 or y_arr.ndim != 2:
        raise ValueError("u_tb and y_tb must both have shape (T,B)")
    if u_arr.shape != y_arr.shape:
        raise ValueError("u_tb and y_tb must have matching shape (T,B)")

    t_count = int(u_arr.shape[0])
    b_count = int(u_arr.shape[1])
    split_counts = tuple(int(v) for v in cfg["train_val_test_split"])
    if sum(split_counts) != b_count:
        raise ValueError(f"train_val_test_split must sum to B={b_count}, got {split_counts}")

    train_idx, val_idx, test_idx = features.split_batch_indices(
        n_batch=b_count,
        split_counts=split_counts,
        split_seed=int(cfg["split_seed"]),
        shuffle=bool(cfg["shuffle_split"]),
    )

    # u_scaled_tb: shape (T,B). This is the only raw signal used by the direct MLP.
    u_scaled_tb = scale_u_for_direct_mlp(u_arr, cfg)

    # x_raw_btd: shape (B,T,D), D = u_delay_steps + 1.
    x_raw_btd = build_u_delay_features(u_scaled_tb, int(cfg["u_delay_steps"]))
    y_all_bt = np.transpose(y_arr, (1, 0)).astype(float)  # (B,T)

    d_count = int(x_raw_btd.shape[2])
    x_train_raw_2d = x_raw_btd[train_idx].reshape(-1, d_count)  # (B_train*T,D)
    x_mean_d, x_std_d = _norm_stats_2d(x_train_raw_2d, str(cfg["feature_norm_mode"]))

    # x_all_btd: shape (B,T,D), normalized delayed-u feature data.
    x_all_btd = (x_raw_btd - x_mean_d.reshape(1, 1, d_count)) / x_std_d.reshape(1, 1, d_count)
    if cfg.get("x_max", None) is not None:
        x_max = float(cfg["x_max"])
        x_all_btd = np.clip(x_all_btd, -x_max, x_max)

    y_train_1d = y_all_bt[train_idx].reshape(-1)  # (B_train*T,)
    if str(cfg["target_norm_mode"]).lower() == "zscore":
        y_mean = float(np.mean(y_train_1d))
        y_std = float(np.std(y_train_1d))
        if y_std < 1e-8:
            y_std = 1.0
    else:
        y_mean = 0.0
        y_std = 1.0

    def flatten_batches(batch_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Flatten selected trajectory batches for direct-MLP supervised learning.

        Input:
        - batch_idx: ndarray, shape (B_selected,), integer trajectory indices.

        Outputs:
        - X_2d: ndarray, shape (B_selected*T,D), normalized input-delay features.
        - y_1d: ndarray, shape (B_selected*T,), target output samples.

        This helper is baseline-only and does not implement the NFIR lifting
        equations in arXiv:2508.05279v2.
        """
        # X_2d: shape (len(batch_idx)*T,D); y_1d: shape (len(batch_idx)*T,).
        X_2d = x_all_btd[batch_idx].reshape(-1, d_count)
        y_1d = y_all_bt[batch_idx].reshape(-1)
        return X_2d.astype(np.float32), y_1d.astype(np.float32)

    X_train, y_train = flatten_batches(train_idx)
    X_val, y_val = flatten_batches(val_idx)
    X_test, y_test = flatten_batches(test_idx)

    out: dict[str, Any] = {}
    out["u_tb"] = u_arr
    out["y_tb"] = y_arr
    out["u_scaled_tb"] = u_scaled_tb
    out["x_raw_btd"] = x_raw_btd
    out["x_all_btd"] = x_all_btd.astype(np.float32)
    out["y_all_bt"] = y_all_bt.astype(np.float32)
    out["X_train"] = X_train
    out["y_train"] = y_train
    out["X_val"] = X_val
    out["y_val"] = y_val
    out["X_test"] = X_test
    out["y_test"] = y_test
    out["train_idx"] = train_idx
    out["val_idx"] = val_idx
    out["test_idx"] = test_idx
    out["x_mean_d"] = x_mean_d.astype(float)
    out["x_std_d"] = x_std_d.astype(float)
    out["y_mean"] = y_mean
    out["y_std"] = y_std
    return out


def target_to_train_units(y_original: np.ndarray, y_mean: float, y_std: float) -> np.ndarray:
    """Convert original y values to normalized training units."""
    return (np.asarray(y_original, dtype=float) - float(y_mean)) / float(y_std)


def target_to_original_units(y_train_units: np.ndarray, y_mean: float, y_std: float) -> np.ndarray:
    """Convert normalized model predictions back to original y units."""
    return np.asarray(y_train_units, dtype=float) * float(y_std) + float(y_mean)


def compute_split_mse(y_hat_tb: np.ndarray, y_tb: np.ndarray, idx: np.ndarray) -> float:
    """Compute MSE on selected batch indices."""
    err_tb = np.asarray(y_hat_tb, dtype=float)[:, idx] - np.asarray(y_tb, dtype=float)[:, idx]
    return float(np.mean(err_tb * err_tb))


def save_direct_mlp_outputs(result: dict[str, Any], out_dir: str | Path, run_name: str) -> tuple[Path, Path]:
    """
    Save direct MLP result to PKL and MAT.

    Required result dimensions:
    - u_tb, y_tb, y_hat_tb: shape (T,B)
    - x_all_btd: shape (B,T,D)
    - split indices: shape (n_split,)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / f"{run_name}.pkl"
    mat_path = out_dir / f"{run_name}_train.mat"

    pkl_data = dict(result)
    with pkl_path.open("wb") as f:
        pickle.dump(pkl_data, f)

    tr = np.asarray(result["split_train_idx"], dtype=int)
    va = np.asarray(result["split_val_idx"], dtype=int)
    te = np.asarray(result["split_test_idx"], dtype=int)

    y_hat_tb = np.asarray(result["y_hat_tb"], dtype=float)
    y_tb = np.asarray(result["y_tb"], dtype=float)

    data_mat: dict[str, Any] = {}
    data_mat["schema_version"] = str(result["schema_version"])
    data_mat["mode"] = str(result["mode"])
    data_mat["timestamp_utc"] = str(result["timestamp_utc"])
    data_mat["cfg"] = dict(result["cfg"])
    if data_mat["cfg"].get("x_max", None) is None:
        data_mat["cfg"]["x_max"] = ""

    data_mat["u_tb"] = np.asarray(result["u_tb"], dtype=float)
    data_mat["u_scaled_tb"] = np.asarray(result["u_scaled_tb"], dtype=float)
    data_mat["y_tb"] = y_tb
    data_mat["y_hat_tb"] = y_hat_tb
    data_mat["x_all_btd"] = np.asarray(result["x_all_btd"], dtype=float)
    data_mat["x_mean_d"] = np.asarray(result["x_mean_d"], dtype=float)
    data_mat["x_std_d"] = np.asarray(result["x_std_d"], dtype=float)
    data_mat["y_mean"] = np.array([[float(result["y_mean"])]], dtype=float)
    data_mat["y_std"] = np.array([[float(result["y_std"])]], dtype=float)

    data_mat["y_pre_train_batch"] = y_hat_tb[:, tr]
    data_mat["y_train_batch"] = y_tb[:, tr]
    data_mat["y_pre_val_batch"] = y_hat_tb[:, va]
    data_mat["y_val_batch"] = y_tb[:, va]
    data_mat["y_pre_test_batch"] = y_hat_tb[:, te]
    data_mat["y_test_batch"] = y_tb[:, te]

    data_mat["train_mse"] = np.array([[float(result["train_mse"])]], dtype=float)
    data_mat["val_mse"] = np.array([[float(result["val_mse"])]], dtype=float)
    data_mat["test_mse"] = np.array([[float(result["test_mse"])]], dtype=float)

    data_mat["split_train_idx"] = tr.astype(np.int32)
    data_mat["split_val_idx"] = va.astype(np.int32)
    data_mat["split_test_idx"] = te.astype(np.int32)
    data_mat["split_train_idx_1based"] = tr.astype(np.int32) + 1
    data_mat["split_val_idx_1based"] = va.astype(np.int32) + 1
    data_mat["split_test_idx_1based"] = te.astype(np.int32) + 1

    if "history_epoch" in result:
        data_mat["history_epoch"] = np.asarray(result["history_epoch"])
    if "history_train_loss" in result:
        data_mat["history_train_loss"] = np.asarray(result["history_train_loss"])
    if "history_val_loss" in result:
        data_mat["history_val_loss"] = np.asarray(result["history_val_loss"])
    if "history_test_loss" in result:
        data_mat["history_test_loss"] = np.asarray(result["history_test_loss"])
    if "loss_curve" in result:
        data_mat["loss_curve"] = np.asarray(result["loss_curve"])

    top_key = f"{run_name}_train"
    scipy.io.savemat(str(mat_path), {top_key: data_mat})
    return pkl_path, mat_path


def load_direct_mlp_mat_data(mat_path: str | Path) -> dict[str, Any]:
    """Load MAT data and keep only u_tb/y_tb for direct MLP."""
    loaded = io_data.load_training_mat(mat_path)
    return {
        "u_tb": np.asarray(loaded["u_tb"], dtype=float),
        "y_tb": np.asarray(loaded["y_tb"], dtype=float),
        "source_mat_path": str(loaded["source_mat_path"]),
    }
