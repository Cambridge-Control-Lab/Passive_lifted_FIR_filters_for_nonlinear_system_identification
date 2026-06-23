"""
direct_u_mlp_torch_core.py

Hand-coded PyTorch direct u -> y MLP baseline.
This is a standard MLP with no passive lifted FIR/NFIR structure from
arXiv:2508.05279v2. It is kept as an Exp1 comparison model.

Flow map:
1. ``direct_u_mlp_data`` builds delayed input features X_2d, shape (N,D), and
   normalized targets y_n1, shape (N,1).
2. ``DirectTorchMLP`` maps X_2d directly to y_n1 with a feedforward MLP.
3. The training loop minimizes ordinary MSE and saves baseline predictions.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import direct_u_mlp_data as data_core
from . import theta_N_core as step1_core
class DirectTorchMLP(nn.Module):
    """
    Standard direct MLP.

    Input:
    - x_bd: torch.Tensor, shape (N,D)
      N = number of samples, D = u_delay_steps + 1

    Output:
    - y_b1: torch.Tensor, shape (N,1)
    """

    def __init__(self, input_dim: int, hidden_dims: tuple[int, int]) -> None:
        """
        Initialize the direct baseline MLP.

        Inputs:
        - input_dim: scalar int D, number of delayed input features.
        - hidden_dims: tuple[int,int], widths of the two hidden layers.

        This constructor belongs to the baseline path only. It does not build
        theta_N lifting functions or theta_G FIR filters.
        """
        super().__init__()
        input_dim = int(input_dim)
        h1 = int(hidden_dims[0])
        h2 = int(hidden_dims[1])
        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)

    def forward(self, x_nd: torch.Tensor) -> torch.Tensor:
        """Forward pass: x_nd shape (N,D), return y_n1 shape (N,1)."""
        h1 = torch.tanh(self.fc1(x_nd))
        h2 = torch.tanh(self.fc2(h1))
        y_n1 = self.fc3(h2)
        return y_n1


def build_default_config() -> dict[str, Any]:
    """Build default config for PyTorch direct u -> y MLP."""
    cfg = data_core.build_default_data_config()
    cfg["run_name"] = "run_direct_u_mlp_torch"
    cfg["mode"] = "direct_u_mlp_torch"
    cfg["hidden_layer_sizes"] = (256, 256)
    cfg["learning_rate"] = 1e-3
    cfg["weight_decay"] = 1e-6
    cfg["max_epochs"] = 2000
    cfg["batch_size"] = 4096
    cfg["grad_clip_norm"] = 5.0
    cfg["early_stopping"] = True
    cfg["early_stopping_patience"] = 300
    cfg["early_stopping_min_delta"] = 1e-8
    cfg["verbose"] = True
    cfg["log_every"] = 50
    cfg["device"] = "mps"
    return cfg


def _validate_torch_config(cfg: dict[str, Any]) -> None:
    """Validate important PyTorch direct-MLP config values."""
    hidden = cfg["hidden_layer_sizes"]
    if not isinstance(hidden, tuple) or len(hidden) != 2:
        raise ValueError("hidden_layer_sizes must be a tuple of length 2")
    if int(hidden[0]) < 1 or int(hidden[1]) < 1:
        raise ValueError("hidden layer sizes must be >= 1")
    if float(cfg["learning_rate"]) <= 0.0:
        raise ValueError("learning_rate must be > 0")
    if float(cfg["weight_decay"]) < 0.0:
        raise ValueError("weight_decay must be >= 0")
    if int(cfg["max_epochs"]) < 1:
        raise ValueError("max_epochs must be >= 1")
    if int(cfg["batch_size"]) < 1:
        raise ValueError("batch_size must be >= 1")


def _eval_mse(model: nn.Module, X_nd: torch.Tensor, y_n1: torch.Tensor) -> float:
    """Evaluate normalized-space MSE for tensors X_nd shape (N,D), y_n1 shape (N,1)."""
    model.eval()
    with torch.no_grad():
        y_hat_n1 = model(X_nd)
        loss = F.mse_loss(y_hat_n1, y_n1)
    return float(loss.detach().cpu().item())


def train_direct_u_mlp(u_tb: np.ndarray, y_tb: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    """Train PyTorch direct u -> y MLP."""
    _validate_torch_config(cfg)
    np.random.seed(int(cfg["random_seed"]))
    torch.manual_seed(int(cfg["random_seed"]))

    prepared = data_core.split_direct_mlp_data(u_tb=u_tb, y_tb=y_tb, cfg=cfg)
    device = step1_core._resolve_runtime_device(str(cfg["device"]))

    # X_*: shape (N,D). y_*_train_units: shape (N,1).
    X_train = prepared["X_train"]
    X_val = prepared["X_val"]
    X_test = prepared["X_test"]
    y_train_units = data_core.target_to_train_units(prepared["y_train"], prepared["y_mean"], prepared["y_std"])
    y_val_units = data_core.target_to_train_units(prepared["y_val"], prepared["y_mean"], prepared["y_std"])
    y_test_units = data_core.target_to_train_units(prepared["y_test"], prepared["y_mean"], prepared["y_std"])

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train_units.reshape(-1, 1), dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val_units.reshape(-1, 1), dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test_units.reshape(-1, 1), dtype=torch.float32, device=device)

    d_count = int(X_train.shape[1])
    model = DirectTorchMLP(
        input_dim=d_count,
        hidden_dims=tuple(int(v) for v in cfg["hidden_layer_sizes"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val = np.inf
    best_epoch = -1
    bad_count = 0
    history_epoch: list[int] = []
    history_train: list[float] = []
    history_val: list[float] = []
    history_test: list[float] = []

    n_train = int(X_train_t.shape[0])
    batch_size = min(int(cfg["batch_size"]), n_train)

    if bool(cfg["verbose"]):
        print(f"[direct-torch] device={device} D={d_count} N_train={n_train}")

    train_start_time = time.time()
    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        model.train()
        rng_epoch = np.random.default_rng(int(cfg["random_seed"]) + epoch)
        perm = rng_epoch.permutation(n_train)

        for start in range(0, n_train, batch_size):
            idx_np = perm[start:start + batch_size]
            idx_t = torch.tensor(idx_np, dtype=torch.long, device=device)
            xb = X_train_t[idx_t]
            yb = y_train_t[idx_t]

            optimizer.zero_grad(set_to_none=True)
            y_hat = model(xb)
            loss = F.mse_loss(y_hat, yb)
            loss.backward()
            if cfg.get("grad_clip_norm", None) is not None:
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip_norm"]))
            optimizer.step()

        train_loss = _eval_mse(model, X_train_t, y_train_t)
        val_loss = _eval_mse(model, X_val_t, y_val_t)
        test_loss = _eval_mse(model, X_test_t, y_test_t)
        history_epoch.append(epoch)
        history_train.append(train_loss)
        history_val.append(val_loss)
        history_test.append(test_loss)

        improved = val_loss < best_val - float(cfg["early_stopping_min_delta"])
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_count = 0
        else:
            bad_count += 1

        if bool(cfg["verbose"]):
            should_log = epoch == 1 or epoch % int(cfg["log_every"]) == 0 or improved
            if should_log:
                print(
                    f"[direct-torch] epoch {epoch:4d}/{cfg['max_epochs']} "
                    f"train={train_loss:.6e} val={val_loss:.6e} test={test_loss:.6e} "
                    f"best_val={best_val:.6e} @ {best_epoch}"
                )

        if bool(cfg["early_stopping"]) and bad_count >= int(cfg["early_stopping_patience"]):
            if bool(cfg["verbose"]):
                print(f"[direct-torch] early stopping at epoch {epoch}")
            break

    train_time_sec = time.time() - train_start_time

    model.load_state_dict(best_state)
    model.eval()

    x_all_btd = np.asarray(prepared["x_all_btd"], dtype=np.float32)  # (B,T,D)
    b_count = int(x_all_btd.shape[0])
    t_count = int(x_all_btd.shape[1])
    X_all_2d = x_all_btd.reshape(-1, d_count)  # (B*T,D)
    X_all_t = torch.tensor(X_all_2d, dtype=torch.float32, device=device)
    with torch.no_grad():
        y_hat_units_1d = model(X_all_t).detach().cpu().numpy().reshape(-1)
    y_hat_1d = data_core.target_to_original_units(y_hat_units_1d, prepared["y_mean"], prepared["y_std"])
    y_hat_bt = y_hat_1d.reshape(b_count, t_count)  # (B,T)
    y_hat_tb = np.transpose(y_hat_bt, (1, 0))  # (T,B)

    train_mse = data_core.compute_split_mse(y_hat_tb, prepared["y_tb"], prepared["train_idx"])
    val_mse = data_core.compute_split_mse(y_hat_tb, prepared["y_tb"], prepared["val_idx"])
    test_mse = data_core.compute_split_mse(y_hat_tb, prepared["y_tb"], prepared["test_idx"])

    result: dict[str, Any] = {}
    result["schema_version"] = str(cfg["schema_version"])
    result["mode"] = "direct_u_mlp_torch"
    result["timestamp_utc"] = data_core.utc_now_iso()
    result["cfg"] = dict(cfg)
    result["device"] = str(device)
    # Store model parameters on CPU so the PKL can be loaded without requiring
    # the same runtime backend. Each tensor has its usual PyTorch shape.
    model_state_cpu = {}
    for key_name, state_value in model.state_dict().items():
        model_state_cpu[key_name] = state_value.detach().cpu()
    result["model_state_dict"] = model_state_cpu
    result["u_tb"] = prepared["u_tb"]
    result["u_scaled_tb"] = prepared["u_scaled_tb"]
    result["y_tb"] = prepared["y_tb"]
    result["y_hat_tb"] = y_hat_tb
    result["x_all_btd"] = prepared["x_all_btd"]
    result["x_mean_d"] = prepared["x_mean_d"]
    result["x_std_d"] = prepared["x_std_d"]
    result["y_mean"] = prepared["y_mean"]
    result["y_std"] = prepared["y_std"]
    result["split_train_idx"] = prepared["train_idx"]
    result["split_val_idx"] = prepared["val_idx"]
    result["split_test_idx"] = prepared["test_idx"]
    result["train_mse"] = train_mse
    result["val_mse"] = val_mse
    result["test_mse"] = test_mse
    result["best_epoch"] = int(best_epoch)
    result["best_val_loss"] = float(best_val)
    result["mlp_train_time_sec"] = float(train_time_sec)
    result["nn_train_time_sec"] = float(train_time_sec)
    result["history_epoch"] = np.asarray(history_epoch, dtype=int)
    result["history_train_loss"] = np.asarray(history_train, dtype=float)
    result["history_val_loss"] = np.asarray(history_val, dtype=float)
    result["history_test_loss"] = np.asarray(history_test, dtype=float)
    return result


def run_from_mat_file(mat_path: str | Path, out_dir: str | Path, run_name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Load MAT data, train PyTorch direct MLP, and save outputs."""
    loaded = data_core.load_direct_mlp_mat_data(mat_path)
    result = train_direct_u_mlp(loaded["u_tb"], loaded["y_tb"], cfg)
    result["source_mat_path"] = str(loaded["source_mat_path"])
    pkl_path, mat_out_path = data_core.save_direct_mlp_outputs(result, out_dir, run_name)
    result["pkl_path"] = str(pkl_path)
    result["mat_path"] = str(mat_out_path)
    return result
