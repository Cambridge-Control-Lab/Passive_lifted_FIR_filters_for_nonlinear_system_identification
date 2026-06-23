from __future__ import annotations

import copy
import pickle
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import features
from . import io_data

import time 


def _apply_mlp_activation(x: torch.Tensor, name: str) -> torch.Tensor:
    """Apply one supported MLP activation to tensor x with any shape."""
    name_text = str(name).strip().lower()
    if name_text == "tanh":
        return torch.tanh(x)
    if name_text == "relu":
        return torch.relu(x)
    if name_text == "gelu":
        return F.gelu(x)
    if name_text == "sigmoid":
        return torch.sigmoid(x)
    if name_text == "identity":
        return x
    raise ValueError(f"Unsupported MLP activation: {name}")


class SharedMLP(nn.Module):
    """
      Shared MLP lifting model used by NFIR Step1.

      SharedMLP(nn.Module)
      - means this class is inheriting nn.Module class

      Input tensor dimension:
      - x_btf: torch.Tensor, shape (B, T, F)
        B = number of trajectories in a mini-batch
        T = number of time steps
        F = number of extended scheduling features

      Output tensor dimension:
      - k_btj: torch.Tensor, shape (B, T, J)
        J = number of NFIR branches

      Forward structure:
      - t axis: time index, length T
      - b axis: trajectory index, length B
      - Apply same MLP to each (b,t) feature vector independently.
      - Final head output uses cfg-selected output activation.
    """

    def __init__(self, input_dim: int, n_branch: int, 
                 hidden_dims: tuple[int, int],
                 hidden_activation: str = "tanh",
                 output_activation: str = "tanh") -> None:
        """
          Init function for the class SharedMLP

          input_dim: 
          - dim of NN input. So it should be F, the dime of feature
         
          n_branch:
          - dim of NN output which is number of NFIR branches, so call it n_branch
          
          hidden_dims:
          - dim of hidden layers. We assume two hidden layers in total

          hidden_activation:
          - scalar string activation for hidden layers, e.g. "tanh" or "gelu"

          output_activation:
          - scalar string activation for output layer, e.g. "tanh" or "identity"
        """
        # Since we inherit the class nn.Module, we need to init that class
        super().__init__()

        # Convert relevant things to int
        input_dim = int(input_dim)
        n_branch = int(n_branch)

        # Validate key dimensions.
        if input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        if n_branch < 1:
            raise ValueError("n_branch must be >= 1")
        
        # Read two hidden layer sizes/dimensions.
        h1 = int(hidden_dims[0])
        h2 = int(hidden_dims[1])

        # Validate hidden layer sizes/dimension.
        if h1 < 1 or h2 < 1:
            raise ValueError("hidden layer dimension must be >= 1")
        
        # Activation names are scalar strings; dimensions of tensors are unchanged by activation.
        self.hidden_activation = str(hidden_activation).strip().lower()
        self.output_activation = str(output_activation).strip().lower()

        # Define the hidden layer structure
        self.fc1 = nn.Linear(input_dim, h1) # first hidden layer, assume bias is also learnt
        self.fc2 = nn.Linear(h1, h2) # second hidden layer
        self.fc3 = nn.Linear(h2, n_branch) # output layer
        
    def forward(self, x_btf: torch.Tensor) -> torch.Tensor:
        """
          Forward pass: given fixed NN para and NN input, gives NN output

          Usage:
          - Output the class, once you have created an instance of this class, e.g., call it model
          Then model(x_data) automatically returns the output of this function.
          Direct calls to model.forward(x_data) are not needed.
          Input:
          - x_btf: (B, T, F), extended scheduling signal/feature

          Output:
          - k_btj: (B, T, J), input of each NFIR branch. 
        """

        # Validate NN input dimension
        if x_btf.ndim != 3:
            raise ValueError(f"x_btf must be 3D, got {tuple(x_btf.shape)}")

        # Read dimensions.
        b_count = int(x_btf.shape[0])
        t_count = int(x_btf.shape[1])
        f_count = int(x_btf.shape[2])

        # Flatten (B,T,F) -> (B*T,F) to feed linear layers.
        # Evaluate all batches in one vectorized pass.
        x_flat = x_btf.reshape(b_count * t_count, f_count)

        # Layer 1 + selected hidden activation.
        """
          y = f(Wu + b). 
          where W is the weights matrix to learn.
                b is the bias vector to learn. 
                f is a fixed static nonlinearity selected by cfg.
        """
        h1 = self.fc1(x_flat)             # shape (B*T,h1)
        h1 = _apply_mlp_activation(h1, self.hidden_activation)

        # second hidden layer
        h2 = self.fc2(h1)                 # shape (B*T,h2)
        h2 = _apply_mlp_activation(h2, self.hidden_activation)

        # output layer
        y_flat = self.fc3(h2)             # shape (B*T,J)
        y_flat = _apply_mlp_activation(y_flat, self.output_activation)

        # Reshape back to (B, T, J).
        y_btj = y_flat.reshape(b_count, t_count, -1)

        # Return branch gains.
        return y_btj


def build_default_config() -> dict[str,Any]:
    """
      Build default config matching current old Step1 run path.

      Output:
      - cfg: dict with scalar fields only.

      Important config dimensions:
      - train_val_test_split: tuple of 3 integers over batch axis.
      - hidden_dims: tuple of 2 hidden widths.
      - active_dims: tuple of active base scheduling dimensions.
      - delay_steps_by_dim: dict mapping base dim -> delay count.

      Python dict facts:
      - e.g. cfg = {"gg_name": "gg",
                    "gg_age" :  10"}.  
        This is dictionary in python. "gg_name" is a key, "gg" is the value.
        So  -> dict[str,Any]: means the keys are strings, 
        and the values can be of any type (e.g., int, float, str, list, etc.).
    """
      
    cfg = {}

    # Run/export naming.
    cfg["run_name"] = "run0"
    cfg["schema_version"] = "nfir8e_step1_v1"
    cfg["mode"] = "step1_nn"

    # Model/FIR dimensions.
    cfg["n_branch"] = 6 # num of branches in the FIR bank.
    cfg["m_fir"] = 100   # FIR length.
    cfg["dt"] = 0.02     # time step size in seconds.

    # FIR deterministic init settings. Since we need to init FIR then train NN
    cfg["fir_seed"] = 7       # Random seed for deterministic jitter when generating FIR parameters. Same seed => same FIR bank.
    cfg["fir_tau_min"] = 0.12 # 
    cfg["fir_tau_max"] = 0.45 # Range of branch time constants tau_j (seconds).
                              # Smaller tau = faster decay; larger tau = slower decay.
    cfg["fir_gain_min"] = 0.6 # 
    cfg["fir_gain_max"] = 1.4 # Range of branch gains gain_j (initial amplitude scale).
    cfg["fir_source_type"] = "exponential" # scalar string: FIR source type, "exponential" or "step2_pkl".
    cfg["fir_source_step2_pkl"] = "" # path string to Step2 pkl when fir_source_type is "step2_pkl", else empty string.
    """
      g_j[k] = fir_gain_j * exp ( - k*dt/fir_tau), k = 0,...,M-1.
      j = 0,..., n_branch - 1
    """
    # Feature construction settings.
    cfg["active_dims"] = (0, 1)  # scheduling signal has many dimensionals. Here selet which dim is used. 
    cfg["delay_steps_by_dim"] = {} # 0:5 means the 0-th dimension has 5 steps of delay. So (p0(t), p0(t-1), ..., p0(t-5)) are used. 
    cfg["scale_io_by_20"] = False # A rubbish normalisation to scale io training data so that all are between -1 and 1. To improve later. 

    # Feature normalization mode: whether normalise features. 
    cfg["feature_norm_mode"] = "none"
    cfg["x_max"] = None  # None: no clipping; float > 0 clips normalized feature x to [-x_max, x_max].

    # Closed-loop scaling mode for causal p_7tb rebuild.
    cfg["fixed_uy_scale"] = True  # bool scalar: True uses fixed u/y scales, False uses dynamic running scales.
    cfg["u_scale_fixed"] = 22.0  # float scalar: fixed scale for channel p0 from u(t).
    cfg["y_scale_fixed"] = 31.471743603975618  # float scalar: fixed scale for channel p1 from y_hat(t).
    cfg["uy_scale_method"] = "divide"  # scalar string: "softsign" for x/sqrt(x^2+s^2), or "divide" for x/s.
    cfg["u_max_after_scale"] = 100.0  # float scalar: clip bound for scaled p0 channel.
    cfg["y_max_after_scale"] = 100.0  # float scalar: clip bound for scaled p1 channel.
    cfg["rebuild_p7_from_uy"] = False  # bool scalar: if True, rebuild open-loop p_7tb from loaded u_tb,y_tb.

    # NN network structure.
    cfg["hidden_dims"] = (128, 128) # num of neurons in hidden layer 1 and 2
    cfg["mlp_hidden_activation"] = "tanh"  # scalar string: activation for hidden layers.
    cfg["mlp_output_activation"] = "tanh"  # scalar string: activation for output layer k_j(t).

    # NN training settings. To add annotation later. 
    cfg["learning_rate"] = 1e-3 # gradient descent update step size. 
                                # ith AdamW, the actual per-parameter update is adaptive, 
                                # but this value is still the main global scale.
    cfg["weight_decay"] = 1e-5 #L2-style regularization in AdamW. It gently pushes weights toward smaller values to reduce overfitting.
    cfg["max_epochs"] = 300 
    cfg["batch_size"] = 4 # How many batches of data in each mini-batch.
    cfg["grad_clip_norm"] = 5.0 # If total gradient norm is bigger than 5.0, gradients are scaled down to norm 5.0 before optimizer step, preventing unstable huge updates.

    # Train, val and test split setting
    cfg["train_val_test_split"] = (16, 2, 2)
    cfg["split_seed"] = 42
    cfg["shuffle_split"] = False # Decide, if 20 batches of data are provided with 16 used for training
                                # then false means we just use the first 16 for training, next 2 for val, last 2 for test
                                # then true means we shuffle the order of these 20 batches, then take first 16 of the shuffled batch order for training
    cfg["deterministic_epoch_shuffle"] = True # Once 16 batches of data is decided, for each epoch, we need 
                                              # to break the 16 batches to mini-batches, each mimi-batch
                                              # has cfg["batch_size"] batches of data. 
                                              # However, for each epoch, we ideally want the content of mini-batches to be different
                                              # so if this is true, then we shuffle these 16 batches before spliting it to mini-batches 

    # Random seed for model init and numpy seeding.
    cfg["random_seed"] = 42

    # NN training early stopping settings. To add annotation later. 
    cfg["early_stopping"] = True
    cfg["early_stopping_patience"] = 50 # Stop training if validation loss fails to improve for 50 consecutive epochs.
    cfg["early_stopping_min_delta"] = 1e-6 # Defines what counts as an “improvement”: new val loss must be at least 1e-6 lower than current best.
    # Logging settings.
    cfg["verbose"] = True # Means in debug mode, print out things during training. 
    cfg["log_every"] = 10  # how often to print out.
    cfg["save_full_diagnostics"] = False # whether save all details 

    # Device settings.
    # cfg["device"] = "cpu" 
    cfg["device"] = "mps" # use apple silicon GPU if mps 
    
    """BPTT paras
        bptt_finetune_enable:
        False keeps old behavior.
        True runs the closed-loop fine-tuning phase after normal Step1 training.

        bptt_max_epochs:
        Number of fine-tuning epochs.
        If zero, no BPTT training should run.

        bptt_learning_rate:
        Usually smaller than open-loop learning_rate.
        Recommended first value: 1e-4.

        bptt_batch_size:
        Number of trajectories/batches in each BPTT mini-batch.
        Smaller than open-loop batch_size is usually safer because BPTT stores a larger computation graph.

        bptt_early_stopping:
        Whether to stop BPTT based on closed-loop validation loss.

        bptt_early_stopping_patience:
        How many BPTT epochs to wait without validation improvement.

        bptt_grad_clip_norm:
        Gradient clipping for closed-loop rollout training.
        This matters because gradients are propagated through time.

        If the closed-loop loss explodes, try in this order:
            1. reduce bptt_learning_rate to 3e-5
            2. reduce bptt_grad_clip_norm to 0.5
            3. reduce bptt_max_epochs
            4. start from a smaller hidden_dims model
        
        If train closed-loop loss improves but validation closed-loop loss worsens:
            1. reduce bptt_max_epochs
            2. increase weight_decay
            3. use smaller hidden_dims
            4. keep BPTT as a short fine-tuning phase, not long training


    """
    cfg["bptt_finetune_enable"] = False  
    cfg["bptt_max_epochs"] = 70
    cfg["bptt_learning_rate"] = 5e-5
    cfg["bptt_batch_size"] = 4
    cfg["bptt_early_stopping"] = True
    cfg["bptt_early_stopping_patience"] = 40
    cfg["bptt_grad_clip_norm"] = 1.0

    # BPTT learning-rate scheduler.
    # "none" keeps the old fixed-learning-rate behavior.
    cfg["bptt_lr_scheduler"] = "none"  # "none", "plateau", "exponential", "cosine", "warmup_cosine", "onecycle"
    cfg["bptt_lr_decay_factor"] = 0.5  # scalar, used by plateau.
    cfg["bptt_lr_decay_patience"] = 5  # scalar int, used by plateau.
    cfg["bptt_lr_gamma"] = 0.95  # scalar, used by exponential.
    cfg["bptt_min_learning_rate"] = 1e-6  # scalar, used by plateau/cosine/warmup_cosine.
    cfg["bptt_warmup_epochs"] = 5  # scalar int, used by warmup_cosine.
    cfg["bptt_onecycle_pct_start"] = 0.3  # scalar, fraction of updates spent increasing LR.
    cfg["bptt_onecycle_div_factor"] = 25.0  # scalar, initial LR = max_lr / div_factor.
    cfg["bptt_onecycle_final_div_factor"] = 1e4 # scalar, final LR = initial LR / final_div_factor.
    
    cfg["skip_open_loop_training"] = False # If this is True, then no open loop training at all.
    # Return ready config.
    return cfg


def _validate_config(cfg: dict[str, Any]) -> None:
    """
    Validate key configuration entries for Step1 minimal path.

    Input:
    - cfg: config dict

    Output:
    - none (raises ValueError if invalid)
    """
    # Check key presence first so error messages are clear.
    required_keys = [
        "n_branch",
        "m_fir",
        "learning_rate",
        "max_epochs",
        "batch_size",
        "train_val_test_split",
        "hidden_dims",
        "active_dims",
        "delay_steps_by_dim",
        "fir_tau_min",
        "fir_tau_max",
        "fir_gain_min",
        "fir_gain_max",
        "fir_source_type",
        "fir_source_step2_pkl",
        "fixed_uy_scale",
        "u_scale_fixed",
        "y_scale_fixed",
        "grad_clip_norm",
        "log_every",
        "device",
        "bptt_finetune_enable",
        "bptt_max_epochs",
        "bptt_learning_rate",
        "bptt_batch_size",
        "bptt_early_stopping",
        "bptt_early_stopping_patience",
        "bptt_grad_clip_norm",
        "bptt_lr_scheduler",
        "bptt_lr_decay_factor",
        "bptt_lr_decay_patience",
        "bptt_lr_gamma",
        "bptt_min_learning_rate",
        "bptt_warmup_epochs",
        "bptt_onecycle_pct_start",
        "bptt_onecycle_div_factor",
        "bptt_onecycle_final_div_factor",
        "skip_open_loop_training",
    ]
    for key_name in required_keys:
        if key_name not in cfg:
            raise ValueError(f"Missing required cfg key: {key_name}")

    # Check branch count.
    if int(cfg["n_branch"]) < 1:
        raise ValueError("n_branch must be >= 1")

    # Keep fixed FIR length policy for this rewrite.
    # if int(cfg["m_fir"]) != 100:
    #     raise ValueError("m_fir must be exactly 100 in this minimal rewrite")
    if int(cfg["m_fir"]) < 1: 
        raise ValueError("m_fir must be >= 1")
    # Check learning parameters.
    if float(cfg["learning_rate"]) <= 0.0:
        raise ValueError("learning_rate must be > 0")
    if int(cfg["max_epochs"]) < 1:
        raise ValueError("max_epochs must be >= 1")
    if int(cfg["batch_size"]) < 1:
        raise ValueError("batch_size must be >= 1")

    # Check split tuple.
    split_counts = cfg["train_val_test_split"]
    if len(split_counts) != 3:
        raise ValueError("train_val_test_split must have exactly 3 values")
    n_tr = int(split_counts[0])
    n_va = int(split_counts[1])
    n_te = int(split_counts[2])
    if n_tr < 1 or n_va < 1 or n_te < 1:
        raise ValueError("Each split count must be >= 1 (train, val, test).")

    # Check hidden layer tuple.
    hidden_dims = cfg["hidden_dims"]
    if not isinstance(hidden_dims, tuple):
        raise ValueError("hidden_dims must be a tuple of length 2.")
    if len(hidden_dims) != 2:
        raise ValueError("hidden_dims must have exactly 2 values.")
    if int(hidden_dims[0]) < 1 or int(hidden_dims[1]) < 1:
        raise ValueError("hidden_dims values must be >= 1.")

    hidden_activation_text = str(cfg.get("mlp_hidden_activation", "tanh")).strip().lower()
    if hidden_activation_text not in ("tanh", "relu", "gelu", "sigmoid", "identity"):
        raise ValueError("mlp_hidden_activation must be tanh, relu, gelu, sigmoid, or identity")
    output_activation_text = str(cfg.get("mlp_output_activation", "tanh")).strip().lower()
    if output_activation_text not in ("tanh", "sigmoid", "identity"):
        raise ValueError("mlp_output_activation must be tanh, sigmoid, or identity")

    # Check feature selection dims.
    active_dims = cfg["active_dims"]
    if not isinstance(active_dims, tuple):
        raise ValueError("active_dims must be a tuple.")
    if len(active_dims) < 1:
        raise ValueError("active_dims must contain at least one dimension.")
    for dim_value in active_dims:
        dim_int = int(dim_value)
        if dim_int < 0 or dim_int > 6:
            raise ValueError("active_dims values must be in range [0, 6].")

    # Check delay settings.
    delay_steps = cfg["delay_steps_by_dim"]
    if not isinstance(delay_steps, dict):
        raise ValueError("delay_steps_by_dim must be a dict.")
    for dim_key in delay_steps:
        lag_count = int(delay_steps[dim_key])
        if lag_count < 0:
            raise ValueError("delay_steps_by_dim values must be >= 0.")

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

    # Check FIR range settings.
    tau_min = float(cfg["fir_tau_min"])
    tau_max = float(cfg["fir_tau_max"])
    gain_min = float(cfg["fir_gain_min"])
    gain_max = float(cfg["fir_gain_max"])
    if tau_min <= 0.0 or tau_max <= 0.0:
        raise ValueError("fir_tau_min and fir_tau_max must be > 0.")
    if tau_min > tau_max:
        raise ValueError("fir_tau_min must be <= fir_tau_max.")
    if gain_min <= 0.0 or gain_max <= 0.0:
        raise ValueError("fir_gain_min and fir_gain_max must be > 0.")
    if gain_min > gain_max:
        raise ValueError("fir_gain_min must be <= fir_gain_max.")

    # Check FIR source selector.
    fir_source_type_text = str(cfg["fir_source_type"]).strip().lower() # scalar string: source type text.
    if fir_source_type_text not in ("exponential", "step2_pkl"):
        raise ValueError("fir_source_type must be 'exponential' or 'step2_pkl'")
    if fir_source_type_text == "step2_pkl":
        fir_source_path_raw = cfg.get("fir_source_step2_pkl", "") # raw value expected as scalar string path.
        if not isinstance(fir_source_path_raw, str):
            raise ValueError("fir_source_step2_pkl must be a path string")
        fir_source_path_text = fir_source_path_raw.strip() # scalar string: normalized Step2 pkl path text.
        if len(fir_source_path_text) == 0:
            raise ValueError("fir_source_step2_pkl must be set when fir_source_type='step2_pkl'")

    # Check closed-loop fixed-scale settings.
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

    # Check gradient clipping and logging.
    if cfg["grad_clip_norm"] is not None:
        if float(cfg["grad_clip_norm"]) <= 0.0:
            raise ValueError("grad_clip_norm must be > 0 when provided.")
    if int(cfg["log_every"]) < 1:
        raise ValueError("log_every must be >= 1.")

    # Check BPTT fine-tuning settings once at config validation.
    if not isinstance(cfg["bptt_finetune_enable"], bool):
        raise ValueError("bptt_finetune_enable must be bool")
    if int(cfg["bptt_max_epochs"]) < 0:
        raise ValueError("bptt_max_epochs must be >= 0")
    if float(cfg["bptt_learning_rate"]) <= 0.0:
        raise ValueError("bptt_learning_rate must be > 0")
    if int(cfg["bptt_batch_size"]) < 1:
        raise ValueError("bptt_batch_size must be >= 1")
    if not isinstance(cfg["bptt_early_stopping"], bool):
        raise ValueError("bptt_early_stopping must be bool")
    if int(cfg["bptt_early_stopping_patience"]) < 1:
        raise ValueError("bptt_early_stopping_patience must be >= 1")
    if cfg["bptt_grad_clip_norm"] is not None:
        if float(cfg["bptt_grad_clip_norm"]) <= 0.0:
            raise ValueError("bptt_grad_clip_norm must be > 0 when provided.")
    
    # For validating bptt decay rate
    valid_bptt_lr_schedulers = {
        "none",
        "plateau",
        "exponential",
        "cosine",
        "warmup_cosine",
        "onecycle",
    }
    if str(cfg["bptt_lr_scheduler"]) not in valid_bptt_lr_schedulers:
        raise ValueError("bptt_lr_scheduler must be one of none, plateau, exponential, cosine, warmup_cosine, onecycle")

    if float(cfg["bptt_lr_decay_factor"]) <= 0.0 or float(cfg["bptt_lr_decay_factor"]) >= 1.0:
        raise ValueError("bptt_lr_decay_factor must be between 0 and 1")
    
    if int(cfg["bptt_lr_decay_patience"]) < 1:
        raise ValueError("bptt_lr_decay_patience must be >= 1")
    
    if float(cfg["bptt_lr_gamma"]) <= 0.0 or float(cfg["bptt_lr_gamma"]) > 1.0:
        raise ValueError("bptt_lr_gamma must be in (0, 1]")
    
    if float(cfg["bptt_min_learning_rate"]) <= 0.0:
        raise ValueError("bptt_min_learning_rate must be > 0")
    
    if int(cfg["bptt_warmup_epochs"]) < 0:
        raise ValueError("bptt_warmup_epochs must be >= 0")
    
    if float(cfg["bptt_onecycle_pct_start"]) <= 0.0 or float(cfg["bptt_onecycle_pct_start"]) >= 1.0:
        raise ValueError("bptt_onecycle_pct_start must be between 0 and 1")
    
    if float(cfg["bptt_onecycle_div_factor"]) <= 1.0:
        raise ValueError("bptt_onecycle_div_factor must be > 1")
    
    if float(cfg["bptt_onecycle_final_div_factor"]) <= 1.0:
        raise ValueError("bptt_onecycle_final_div_factor must be > 1")
    
    # Check runtime device selection policy.
    # We keep this strict on purpose:
    # - Only explicit backends are allowed.
    # - No silent fallback is allowed when user requested a specific backend.
    device_text = str(cfg["device"]).strip().lower()
    if device_text not in ("cpu", "mps", "cuda"):
        raise ValueError("device must be one of: cpu, mps, cuda")
    
    if not isinstance(cfg["skip_open_loop_training"], bool):
        raise ValueError("skip_open_loop_training must be bool")


def _load_fir_bank_from_step2_pkl(step2_pkl_path: str) -> np.ndarray:
    """
    Load FIR bank from Step2 pickle.

    Input:
    - step2_pkl_path: path string to Step2 pickle file

    Output:
    - g_bank_jm: np.ndarray, shape (J,M)
    """
    # step2_path_obj: Path scalar, normalized source file path.
    step2_path_obj = Path(step2_pkl_path)
    if not step2_path_obj.exists():
        raise ValueError(f"Step2 pickle file does not exist: {step2_path_obj}")

    # step2_dict: dict[str, Any], top-level object loaded from pickle.
    with step2_path_obj.open("rb") as file_obj:
        step2_dict = pickle.load(file_obj)
    if not isinstance(step2_dict, dict):
        raise ValueError(f"Expected dict in Step2 pickle: {step2_path_obj}")

    if "g_bank" not in step2_dict:
        raise ValueError(f"Step2 pickle missing 'g_bank': {step2_path_obj}")

    # g_bank_jm: np.ndarray, shape (J,M), imported FIR coefficient matrix.
    g_bank_jm = np.asarray(step2_dict["g_bank"], dtype=float)
    if g_bank_jm.ndim != 2:
        raise ValueError(f"Step2 g_bank must be 2D with shape (J,M), got {g_bank_jm.shape}")
    if int(g_bank_jm.shape[0]) < 1 or int(g_bank_jm.shape[1]) < 1:
        raise ValueError(f"Step2 g_bank shape must be positive, got {g_bank_jm.shape}")

    return g_bank_jm


def _resolve_runtime_device(cfg_device: str) -> torch.device:
    """
    Resolve user-configured runtime backend into a concrete torch.device.

    Accepted input values:
    - "cpu": always valid, runs all tensors/model on CPU.
    - "mps": Apple Metal backend for Apple Silicon GPUs (for example M2).
    - "cuda": NVIDIA CUDA backend.

    Output:
    - torch.device object for the selected backend.

    Failure behavior (strict by design):
    - If user requests "mps" and MPS is not available, raise ValueError.
    - If user requests "cuda" and CUDA is not available, raise ValueError.
    - No fallback to CPU is performed when a GPU backend is explicitly requested.
      This avoids accidental slow runs that look like GPU runs.
    """
    backend_text = str(cfg_device).strip().lower()

    # Explicit CPU path.
    if backend_text == "cpu":
        return torch.device("cpu")

    # Explicit Apple Silicon GPU path.
    if backend_text == "mps":
        # is_built means this torch build has MPS support compiled in.
        if not torch.backends.mps.is_built():
            raise ValueError(
                "device='mps' was requested, but this PyTorch build does not include MPS support."
            )
        # is_available means runtime can access MPS on this machine/session.
        if not torch.backends.mps.is_available():
            raise ValueError(
                "device='mps' was requested, but MPS is not available on this machine/runtime."
            )
        return torch.device("mps")

    # Explicit NVIDIA GPU path.
    if backend_text == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "device='cuda' was requested, but CUDA is not available on this machine/runtime."
            )
        return torch.device("cuda")

    # Defensive fallback (should already be prevented by _validate_config).
    raise ValueError("device must be one of: cpu, mps, cuda")


def _sync_device_for_timing(device: torch.device) -> None:
    """
    Synchronize async GPU work before reading wall-clock timers.

    Input:
    - device: torch.device scalar, runtime backend for tensors/model.

    Output:
    - none. CPU has no async device queue to synchronize here.
    """
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _device_report(device: torch.device) -> dict[str, Any]:
    """
    Build a best-effort runtime device report for startup logging.

    Purpose:
    - Provide explicit evidence of backend used (cpu/mps/cuda).
    - Provide backend-specific diagnostics without affecting training logic.

    Output dictionary fields:
    - "backend": "cpu" | "mps" | "cuda"
    - "device": string form of torch.device
    - "details": list[str] with backend-specific info lines

    Reporting policy:
    - CPU: basic backend confirmation only.
    - MPS: memory-oriented stats via torch.mps APIs when available.
      PyTorch does not expose a portable MPS utilization percentage API.
    - CUDA: device name, memory stats, and best-effort utilization percent via nvidia-smi.
      If utilization cannot be queried, report "unavailable" and continue.
    """
    report = {}
    report["backend"] = str(device.type)
    report["device"] = str(device)
    report["details"] = []

    # CPU report is intentionally short.
    if device.type == "cpu":
        report["details"].append("CPU backend selected.")
        return report

    # Apple Silicon (MPS) report.
    if device.type == "mps":
        report["details"].append("MPS backend selected (Apple Metal GPU).")
        if hasattr(torch, "mps"):
            try:
                current_bytes = int(torch.mps.current_allocated_memory())
                report["details"].append(f"mps_current_allocated_memory_bytes={current_bytes}")
            except Exception:
                report["details"].append("mps_current_allocated_memory_bytes=unavailable")
            try:
                driver_bytes = int(torch.mps.driver_allocated_memory())
                report["details"].append(f"mps_driver_allocated_memory_bytes={driver_bytes}")
            except Exception:
                report["details"].append("mps_driver_allocated_memory_bytes=unavailable")
            try:
                recommended_bytes = int(torch.mps.recommended_max_memory())
                report["details"].append(f"mps_recommended_max_memory_bytes={recommended_bytes}")
            except Exception:
                report["details"].append("mps_recommended_max_memory_bytes=unavailable")
        else:
            report["details"].append("mps_memory_stats=unavailable")
        report["details"].append("mps_utilization_percent=unavailable (no standard PyTorch API)")
        return report

    # NVIDIA CUDA report.
    if device.type == "cuda":
        report["details"].append("CUDA backend selected (NVIDIA GPU).")
        try:
            device_index = int(device.index) if device.index is not None else int(torch.cuda.current_device())
            device_name = torch.cuda.get_device_name(device_index)
            report["details"].append(f"cuda_device_index={device_index}")
            report["details"].append(f"cuda_device_name={device_name}")
            allocated_bytes = int(torch.cuda.memory_allocated(device_index))
            reserved_bytes = int(torch.cuda.memory_reserved(device_index))
            report["details"].append(f"cuda_memory_allocated_bytes={allocated_bytes}")
            report["details"].append(f"cuda_memory_reserved_bytes={reserved_bytes}")
        except Exception:
            report["details"].append("cuda_device_info=unavailable")

        # Best-effort utilization percentage using nvidia-smi.
        # This should never fail training if unavailable.
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().splitlines()
                if len(lines) > 0:
                    first_line = lines[0].strip()
                    if len(first_line) > 0:
                        report["details"].append(f"cuda_utilization_percent={first_line}")
                    else:
                        report["details"].append("cuda_utilization_percent=unavailable")
                else:
                    report["details"].append("cuda_utilization_percent=unavailable")
            else:
                report["details"].append("cuda_utilization_percent=unavailable")
        except Exception:
            report["details"].append("cuda_utilization_percent=unavailable")
        return report

    # Fallback for any unexpected backend.
    report["details"].append("backend_details=unavailable")
    return report


def _compute_feature_norm_stats(x_train_btf: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute feature normalization parameters.

    Input:
    - x_train_btf: numpy.ndarray, shape (B_tr, T, F)
    - mode: "none" or "zscore"

    Output:
    - mu_f: numpy.ndarray, shape (F,), float32
    - std_f: numpy.ndarray, shape (F,), float32

    In this rewrite, mode is expected to be "none" to match current script.
    """
    # Normalize mode string.
    mode = str(mode).lower()

    # Read feature dimension.
    feature_count = int(x_train_btf.shape[-1])

    # Mode: none.
    if mode == "none":
        mu_f = np.zeros(feature_count, dtype=np.float32)
        std_f = np.ones(feature_count, dtype=np.float32)
        return mu_f, std_f

    # Mode: zscore (kept for completeness, not used by current script).
    if mode == "zscore":
        x_2d = x_train_btf.reshape(-1, feature_count)
        mu_f = np.mean(x_2d, axis=0).astype(np.float32)
        std_f = np.std(x_2d, axis=0).astype(np.float32)
        for i in range(feature_count):
            if std_f[i] < 1e-6:
                std_f[i] = 1.0
        return mu_f, std_f

    # Unsupported mode.
    raise ValueError(f"Unsupported feature_norm_mode: {mode}")


def causal_per_branch_fir(ku_btj: torch.Tensor, g_jm: torch.Tensor) -> torch.Tensor:
    """
    Apply causal FIR convolution branch-by-branch.

    Input:
    - ku_btj: torch.Tensor, shape (B, T, J)
      Branch-modulated input signal for each branch.
    - g_jm: torch.Tensor, shape (J, M)
      FIR taps per branch.

    Output:
    - v_btj: torch.Tensor, shape (B, T, J)
      Filtered signal per branch.

    Details:
    - For each branch j:
      ku_j has shape (B,T).
      Apply causal 1D convolution with branch taps g_j.
    - Same operation as old implementation:
      left-pad by M-1, then conv1d with time-reversed taps.
    """
    # Validate tensor ranks.
    if ku_btj.ndim != 3:
        raise ValueError("ku_btj must have shape (B,T,J)")
    if g_jm.ndim != 2:
        raise ValueError("g_jm must have shape (J,M)")

    # Read dimensions.
    b_count = int(ku_btj.shape[0])
    t_count = int(ku_btj.shape[1])
    j_count = int(ku_btj.shape[2])
    tap_branch_count = int(g_jm.shape[0])
    m_fir = int(g_jm.shape[1])

    # Validate branch dimension match.
    if j_count != tap_branch_count:
        raise ValueError("ku_btj and g_jm branch dimensions do not match")

    # Container list for each branch output (B,T).
    branch_outputs = []

    # Loop over branches explicitly.
    for j in range(j_count):
        # Extract branch signal: (B,T).
        sig_bt = ku_btj[:, :, j]

        # Reshape for conv1d expected shape: (B, 1, T).
        sig_b1t = sig_bt.contiguous().unsqueeze(1)

        # Left-pad for causal FIR length M.
        sig_b1t_padded = F.pad(sig_b1t, (m_fir - 1, 0))

        # Reverse taps for conv1d correlation -> convolution conversion.
        tap_m = torch.flip(g_jm[j], dims=[0])

        # Reshape taps to conv1d weight shape: (out_ch=1, in_ch=1, kernel=M).
        tap_11m = tap_m.contiguous().reshape(1, 1, m_fir)

        # Perform convolution and remove channel axis.
        out_bt = F.conv1d(sig_b1t_padded, tap_11m).squeeze(1)

        # Append branch output.
        branch_outputs.append(out_bt)

    # Stack branch outputs on last axis to get (B,T,J).
    v_btj = torch.stack(branch_outputs, dim=-1)

    # Return filtered branches.
    return v_btj


def nfir_forward(model: SharedMLP, x_btf: torch.Tensor, u_bt: torch.Tensor, g_jm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
      NFIR forward pass with fixed FIR bank.

      NOTE
      - The output of this function is symbolic expression! 
      It contains the NN parameters variable! 
      In ode language, the output is like 
      ddy = theta1*5*dy + 4 where u = 4. 
      So we have numerical data (input etc) and also symbolic unknown variables!
      Why! Then NN can compute the grad automatically! 

      Input:
      - model: SharedMLP
      - x_btf: (B,T,F) feature data. 
      - u_bt: (B,T)  input
      - g_jm: (J,M)  FIR coef

      Output:
      - y_hat_bt: (B,T)
      - y_branch_btj: (B,T,J)
      - k_btj: (B,T,J)

      Branch equations for each branch j:
      1) k_j(t) = model output
      2) ku_j(t) = k_j(t) * u(t)
      3) v_j(t) = (g_j * ku_j)(t)
      4) y_j(t) = k_j(t) * v_j(t)
      Total output:
      - y_hat(t) = sum_j y_j(t)
    """

    k_btj = model(x_btf) # x_btf are feature data. Pytorch 

    # Compute K(feature) * u with broadcasting over branch axis J.
    ku_btj = k_btj * u_bt.unsqueeze(-1)

    # Do convolution 
    v_btj = causal_per_branch_fir(ku_btj, g_jm)

    # Compute branch outputs.
    y_branch_btj = k_btj * v_btj

    # Sum across branch axis to get total prediction.
    y_hat_bt = torch.sum(y_branch_btj, dim=-1)

    # Return all useful tensors.
    return y_hat_bt, y_branch_btj, k_btj


def _eval_mse(model: SharedMLP, x_btf: torch.Tensor, u_bt: torch.Tensor,
               y_bt: torch.Tensor, g_jm: torch.Tensor) -> float:
    """
      Evaluate MSE loss on one epoch.

      Input:
      - x_btf: (B_split,T,F)
      - u_bt: (B_split,T)
      - y_bt: (B_split,T)
      - g_jm: (J,M)

      Output:
      - mse_value: float
    """
    # Switch to eval mode
    model.eval()

    # Temporarily disable gradient tracking for evaluation.
    with torch.no_grad(): # turn on .no_grad() mode in the following block and turn it off afterwards
        # Forward pass.
        y_hat_bt, _, _ = nfir_forward(model, x_btf, u_bt, g_jm)
        # Compute scalar MSE.
        mse_tensor = F.mse_loss(y_hat_bt, y_bt)

    # Convert scalar tensor to Python float.
    mse_value = float(mse_tensor.detach().cpu().item())
    return mse_value


def _infer_all(
                model: SharedMLP, x_btf: torch.Tensor, u_bt: torch.Tensor, g_jm: torch.Tensor
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Run forward inference for the full dataset.

        Input:
        - x_btf: (B,T,F)
        - u_bt: (B,T)
        - g_jm: (J,M)

        Output:
        - y_hat_bt: numpy.ndarray, shape (B,T)
        - y_branch_btj: numpy.ndarray, shape (B,T,J)
        - k_btj: numpy.ndarray, shape (B,T,J)
    """
    # Switch to eval mode
    model.eval()

    # switch to a mode without grad and close the mode after the following block
    with torch.no_grad():
        y_hat_bt_t, y_branch_btj_t, k_btj_t = nfir_forward(model, x_btf, u_bt, g_jm)
    
    # Move to CPU numpy 
    y_hat_bt = y_hat_bt_t.detach().cpu().numpy()
    y_branch_btj = y_branch_btj_t.detach().cpu().numpy()
    k_btj = k_btj_t.detach().cpu().numpy()

    return y_hat_bt, y_branch_btj, k_btj

def bptt_closed_loop_rollout(
   model: nn.Module,
   u_bt: torch.Tensor,
   y_bt: torch.Tensor,
   g_jm: torch.Tensor,
   feature_map: np.ndarray,
   feature_mean_f: torch.Tensor,
   feature_std_f: torch.Tensor,
   cfg: dict[str, Any]    
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
        Differential version of _infer_all_closed_loop

        Usage:
        - For backprop in time.

        Limitation: 
        - Require cfg["fixed_uy_scale"] = True

        Input:
        - model: nn module. 
        - u_bt: torch.Tensor, shape (B,T)
        - y_bt: torch.Tensor, shape (B,T)
          Target output. It is passed so the BPTT training call keeps the same batch
          bundle, but the rollout itself only uses u_bt and previous predicted y.
        - g_jm: torch.Tensor, shape (J,M)
        - feature_map: np.ndarray, shape (F,2)
        - feature_mean: torch.Tensor, shape (F,)
        - feature_std: torch.Tensor, shape (F,)
        - cfg: dict for step 1

        Output:
        -y_hat_bt_cl: shape (B,T),  closed-loop predicted output for all batches 
        - y_branch_btj_cl: shape (B,T,J) per-branch output contribution.
        - k_btj_cl: shape (B,T,J) NN branch output

        Recursive Logic. At each time step t:   
        - Build p_curr from true input u(t) and previous predicted output y_hat(t-1).
        - Build normalized feature x(t) from p history.
        - Compute k(t) = NN(x(t)).
        - Compute FIR states using k(t)*u(t).
        - Compute y_hat(t).
        - Store y_hat(t) so it affects p(t+1).

        Variable dependency:
        - y_hat(t) -> p(t+1) -> x(t+1) -> k(t+1) -> y_hat(t+1)
        - BPTT trains through this chain.
    """
    if not bool(cfg["fixed_uy_scale"]):
        raise ValueError("BPTT currently requires fixed_uy_scale = True")
    
    u_scale_fixed = cfg["u_scale_fixed"]
    y_scale_fixed = cfg["y_scale_fixed"]
    scale_method_text = str(cfg["uy_scale_method"]).strip().lower()  # scalar string
    if scale_method_text not in ("softsign", "divide"):
        raise ValueError("uy_scale_method must be 'softsign' or 'divide'")
    if bool(cfg.get("scale_io_by_20", False)):
        raise ValueError("BPTT helper requires scale_io_by_20=False")

    B, T = u_bt.shape
    J, M = g_jm.shape
    F = feature_map.shape[0] 
    device = u_bt.device
    dtype = u_bt.dtype
    dt_sec = cfg["dt"]

    """ why adding device=device:
        So make sure it runs on GPU
    """
    nodelay_pre_b7 = torch.zeros(B,7, dtype=dtype, device=device) # previous no-delay p channels.
    int_y_prev_b = torch.zeros(B, dtype=dtype, device=device) # previous integral of predicted y.
    s_hist_bjm  = torch.zeros(B,J,M, dtype=dtype, device=device) # FIR history for each batch and branch.
    p_hist = [] # Python list of length t+1 during rollout. Each item has shape (B,7). At the end, stack to shape (B,T,7) if needed.
    y_hat_list = [] # Python list of length T. Each item has shape (B,). Prediction of NFIR at each time t
    y_branch_list = [] # Python list of length T. Each item has shape (B,J). Prediction of each branch NFIR at each time t
    k_list = [] # Python list of length T. Each item has shape (B,J). Output of the nonlinearity of each branch NFIR at each time t
    # Build p_curr_b7 At Each Time Step. Rules should match _infer_all_closed_loop(...)
   
    u_max_after_scale = cfg["u_max_after_scale"]
    y_max_after_scale = cfg["y_max_after_scale"]
    
    for t_index in range(T):
        p_curr_b7 = torch.zeros(B,7, dtype=dtype, device=device) # Batch at current time t. 
        u_bnow = u_bt[:,t_index] # >>> unow.shape = torch.Size([B]) u_bt.shape =  torch.Size([B, T])

        if scale_method_text == "divide":
            u_scaled_b =  u_bnow / u_scale_fixed
        else:
            u_scaled_b = u_bnow / torch.sqrt(u_bnow * u_bnow + u_scale_fixed * u_scale_fixed)
        u_scaled_b = torch.clamp(u_scaled_b, -u_max_after_scale, u_max_after_scale)
        p_curr_b7[:,0] = u_scaled_b #  p_curr_b7[:,0].shape = torch.Size([B]),  unow.shape = torch.Size([B])
        p_curr_b7[:,1:7] = nodelay_pre_b7[:,1:7]
        p_hist.append(p_curr_b7)

        # At time t, build: x_bf: shape (B,F). x_bf is the feature data at time t.
        x_bf = torch.zeros(B,F, dtype=dtype, device=device)
        """ Idea
            If feature map structure 
             =
            [[0,0]
            [1,0]
            [1,2]
            [3,1]]
            Then it means use dimension 0 and 1 in p(t)
            then use dimension 1 in p(t) with two step delays 
            then use dimension 3 in p(t) with 1 step delays

            So, the Feature data at this time t for one specific batch (e.g, batch idx 1), 
            should be
            x_bf[1,] = [p0(t), p1(t), p1(t-2), p3(t-1)] which is vector of dimension 4. 
            So 
            x_bf[1,] = [p_curr_b7[1, 0]
                        p_curr_b7[1, 1]
                        p_b7_t_mins2[1, 1]
                        p_b7_t_mins1[3, 3]
            p_curr_b7 is just computed above. It is the p(t) 
            p_b7_t_mins2 = p_hist[t_index-2]
            p_b7_t_mins1 = p_hist[t_index-1]
        """
        for f_index in range(F):
            base_dim = int(feature_map[f_index, 0]) # decide p0 or p1 or p3
            lag = int(feature_map[f_index, 1])  # decide t or t-2 or t-1 
            src_t = t_index - lag # if lag = 2, we need to get t_index-2 in p_7tb_cl
            if src_t < 0:
                src_t = 0
            x_val = p_hist[src_t][:,base_dim] #
            x_bf[:,f_index] = x_val

        x_norm_bf = (x_bf - feature_mean_f.reshape(1,F)) / feature_std_f.reshape(1,F) # reshape since x_bf is (B,F)
        x_max = cfg["x_max"]
        if x_max is not None:
            x_norm_bf = torch.clamp(x_norm_bf, -x_max, x_max)  
        k_b1j = model(x_norm_bf.reshape(B,1,F)) # (B,1,J)
        """ Torch trick
            >>> gg = torch.rand((2,1,4))
            >>> gg
            tensor([[[0.4767, 0.4040, 0.1437, 0.5537]],

                    [[0.1528, 0.6514, 0.8994, 0.0813]]])
            >>> gg.shape
            torch.Size([2, 1, 4])
            >>> gg.reshape(2,4)
            tensor([[0.4767, 0.4040, 0.1437, 0.5537],
                    [0.1528, 0.6514, 0.8994, 0.0813]])
        """
        k_bj = k_b1j.reshape(B,J) # (B,J)

        # s_current_bj:  branch FIR inputs at current time.
        """ Torch trick
            >>> gg = torch.rand((2,1))
            >>> gg
            tensor([[0.5243],
                    [0.7128]])
            >>> p_curr_b7
            tensor([[3., 0., 0., 3., 0., 0., 0.],
                    [1., 0., 0., 1., 0., 0., 0.]])
            >>> p_curr_b7 * gg
            tensor([[1.5728, 0.0000, 0.0000, 1.5728, 0.0000, 0.0000, 0.0000],
                    [0.7128, 0.0000, 0.0000, 0.7128, 0.0000, 0.0000, 0.0000]])

            Second trick: 
            >>> s_current_bj.reshape(B,J,1)
            tensor([[[0.1214],
                    [0.8959],
                    [0.7963]],

                    [[0.9028],
                    [0.9388],
                    [0.9460]]])
            >>> s_current_bjm[:,:,0:M-1]
            tensor([[[0.0593, 0.8581, 0.8696, 0.2380],
                    [0.3527, 0.7449, 0.0858, 0.7351],
                    [0.0526, 0.3796, 0.3767, 0.1297]],

                    [[0.2776, 0.1483, 0.8684, 0.1904],
                    [0.4766, 0.7196, 0.8518, 0.2656],
                    [0.2044, 0.4575, 0.5037, 0.7992]]])
            >>> s_current_bjm[:,:,:M-1]
            tensor([[[0.0593, 0.8581, 0.8696, 0.2380],
                    [0.3527, 0.7449, 0.0858, 0.7351],
                    [0.0526, 0.3796, 0.3767, 0.1297]],

                    [[0.2776, 0.1483, 0.8684, 0.1904],
                    [0.4766, 0.7196, 0.8518, 0.2656],
                    [0.2044, 0.4575, 0.5037, 0.7992]]])
            >>> torch.cat([s_current_bj.reshape(B,J,1), s_current_bjm[:,:,0:M-1]
            ... ], dim = 2)
            tensor([[[0.1214, 0.0593, 0.8581, 0.8696, 0.2380],
                    [0.8959, 0.3527, 0.7449, 0.0858, 0.7351],
                    [0.7963, 0.0526, 0.3796, 0.3767, 0.1297]],

                    [[0.9028, 0.2776, 0.1483, 0.8684, 0.1904],
                    [0.9388, 0.4766, 0.7196, 0.8518, 0.2656],
                    [0.9460, 0.2044, 0.4575, 0.5037, 0.7992]]])
        """
        s_current_bj = k_bj * u_bnow.reshape(B,1) # shape (B,J), elementwise product 
        if M > 1:
            s_hist_bjm = torch.cat([s_current_bj.reshape(B, J, 1), s_hist_bjm[:, :, 0:M - 1]], dim=2)# shift history by one step to discard old memory
        else: 
            s_hist_bjm = s_current_bj.reshape(B,J,1) # with no memory/history, history is just current

        # g_jm * s_hist_bjm is elementwise product; summing axis=2 gives branch convolution output.
        g_1jm = g_jm.reshape(1,J,M)
        """ Torch trick
            >>> gg = torch.ones(2,3,4)
            >>> gg
            tensor([[[1., 1., 1., 1.],
                    [1., 1., 1., 1.],
                    [1., 1., 1., 1.]],

                    [[1., 1., 1., 1.],
                    [1., 1., 1., 1.],
                    [1., 1., 1., 1.]]])
            >>> torch.sum(gg, axis=2)
            tensor([[4., 4., 4.],
                    [4., 4., 4.]])
        """
        prob_bjm = s_hist_bjm * g_1jm # shape (B,J,M)
        v_bj = torch.sum(prob_bjm, axis=2)  # shape (B,J).
        y_branch_bj = k_bj * v_bj  # shape (B,J)
        y_hat_b = torch.sum(y_branch_bj, axis=1) # Shape torch.Size([B]), output prediction for all batches at current time t

        y_branch_list.append(y_branch_bj)
        y_hat_list.append(y_hat_b)
        k_list.append(k_bj)

        # Update undelayed channels for next step (delay1 behavior on channels 1..6).
        if scale_method_text == "divide":
            y_scaled_b = y_hat_b / y_scale_fixed
        else:
            y_scaled_b = y_hat_b / torch.sqrt(y_hat_b * y_hat_b + y_scale_fixed * y_scale_fixed)

        y_scaled_b = torch.clamp(y_scaled_b, -y_max_after_scale, y_max_after_scale)
        int_y_now_b = int_y_prev_b + dt_sec * y_hat_b

        nodelay_curr_b7 = torch.zeros(B,7, dtype=dtype, device=device)  # shape (B,7)
        nodelay_curr_b7[:,0] = p_curr_b7[:,0]
        nodelay_curr_b7[:,1] = y_scaled_b
        nodelay_curr_b7[:,2] = torch.sin(int_y_now_b)
        nodelay_curr_b7[:,3] = torch.cos(int_y_now_b)
        nodelay_curr_b7[:,6] = torch.tanh(y_hat_b)

        nodelay_pre_b7 = nodelay_curr_b7
        int_y_prev_b = int_y_now_b
    
    y_hat_tb_cl = torch.stack(y_hat_list, dim=0) # if dim0, then results is (T,B)
    y_hat_bt_cl =y_hat_tb_cl.transpose(0,1) # shape back to (B,T)

    y_branch_tbj_cl = torch.stack(y_branch_list, dim=0) # (T,B,J)
    y_branch_btj_cl = y_branch_tbj_cl.permute(1,0,2) # (B,T,J)

    k_tbj_cl = torch.stack(k_list, dim=0) # (T,B,J)
    k_btj_cl = k_tbj_cl.permute(1,0,2) # (B,T,J)
    
    return y_hat_bt_cl, y_branch_btj_cl, k_btj_cl

def _infer_all_closed_loop(
    model: SharedMLP,
    u_tb: np.ndarray,
    g_jm: np.ndarray,
    feature_map: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    scale_io_by_20: bool,
    dt_sec: float,
    fixed_uy_scale: bool,
    u_scale_fixed: float,
    y_scale_fixed: float,
    uy_scale_method: str,
    u_max_after_scale: float,
    y_max_after_scale: float,
    x_max: float | None,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
        Run closed-loop causal simulation for Step1 model.

        Input:
        - u_tb: np.ndarray, shape (T,B)
        - g_jm: np.ndarray, shape (J,M)
        - feature_map: np.ndarray, shape (F,2)
        - feature_mean: np.ndarray, shape (F,)
        - feature_std: np.ndarray, shape (F,)
        - scale_io_by_20: bool scalar
        - dt_sec: float scalar
        - fixed_uy_scale: bool scalar
        - u_scale_fixed: float scalar
        - y_scale_fixed: float scalar
        - uy_scale_method: string scalar ("softsign" or "divide")
        - u_max_after_scale: float scalar, clip bound for p0 after scaling
        - y_max_after_scale: float scalar, clip bound for p1 after scaling
        - x_max: None or float scalar, clip bound for normalized feature vector

        Output:
        - y_hat_tb_cl: np.ndarray, shape (T,B)
        - y_branch_jtb_cl: np.ndarray, shape (J,T,B)
        - k_jtb_cl: np.ndarray, shape (J,T,B)
        - p_7tb_cl: np.ndarray, shape (7,T,B)
    """
    model.eval()

    u_arr = np.asarray(u_tb, dtype=float)  # shape (T,B)
    g_arr = np.asarray(g_jm, dtype=float)  # shape (J,M)
    fmap = np.asarray(feature_map, dtype=int)  # shape (F,2)
    mu_f = np.asarray(feature_mean, dtype=float).reshape(-1)  # shape (F,)
    std_f = np.asarray(feature_std, dtype=float).reshape(-1)  # shape (F,)

    t_count = int(u_arr.shape[0])  # scalar T
    b_count = int(u_arr.shape[1])  # scalar B
    j_count = int(g_arr.shape[0])  # scalar J
    m_count = int(g_arr.shape[1])  # scalar M
    f_count = int(fmap.shape[0])  # scalar F
    """ why strip lower?
        So "Divide" would pass config validation but code will still use softsign
    """
    scale_method_text = str(uy_scale_method).strip().lower()  # scalar string
    if scale_method_text not in ("softsign", "divide"):
        raise ValueError("uy_scale_method must be 'softsign' or 'divide'")
    u_cap = float(u_max_after_scale)  # scalar clip bound for p0
    y_cap = float(y_max_after_scale)  # scalar clip bound for p1
    if (not np.isfinite(u_cap)) or u_cap <= 0.0:
        raise ValueError("u_max_after_scale must be finite and > 0")
    if (not np.isfinite(y_cap)) or y_cap <= 0.0:
        raise ValueError("y_max_after_scale must be finite and > 0")
    if x_max is None:
        x_cap = None  # no clipping for normalized feature vector x
    else:
        x_cap = float(x_max)  # scalar clip bound for normalized feature vector x

    y_hat_tb_cl = np.zeros((t_count, b_count), dtype=float)  # shape (T,B)
    y_branch_jtb_cl = np.zeros((j_count, t_count, b_count), dtype=float)  # shape (J,T,B)
    k_jtb_cl = np.zeros((j_count, t_count, b_count), dtype=float)  # shape (J,T,B)
    p_7tb_cl = np.zeros((7, t_count, b_count), dtype=float)  # shape (7,T,B)

    for b_index in range(b_count):
        u_t = u_arr[:, b_index]  # shape (T,)
        if bool(fixed_uy_scale):
            u_scale = float(u_scale_fixed)  # scalar fixed u scale
        else:
            u_scale = max(1.1 * float(np.max(np.abs(u_t))), 1e-8)  # scalar dynamic u scale

        y_scale_run = 1e-8  # scalar running scale for causal normalization (dynamic mode only)
        int_y_prev = 0.0  # scalar integral state at previous step
        nodelay_prev_7 = np.zeros(7, dtype=float)  # shape (7,), previous undelayed channels
        s_hist_jm = np.zeros((j_count, m_count), dtype=float)  # shape (J,M), branch FIR input history

        for t_index in range(t_count):
            # p_curr_7: (7,), causal p at current time step.
            p_curr_7 = np.zeros(7, dtype=float)
            u_now = float(u_t[t_index])  # scalar u(t)
            if scale_method_text == "divide":
                p_curr_7[0] = u_now / u_scale
            else:
                p_curr_7[0] = u_now / np.sqrt(u_now * u_now + u_scale * u_scale)
            p_curr_7[0] = float(np.clip(p_curr_7[0], -u_cap, u_cap))
            p_curr_7[1:7] = nodelay_prev_7[1:7]
            p_7tb_cl[:, t_index, b_index] = p_curr_7

            # x_feat_f: (F,), feature vector from causal p history with edge-hold rule. So we now build feature from p(t)
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
                base_dim = int(fmap[f_index, 0]) # decide p0 or p1 or p3
                lag = int(fmap[f_index, 1])  # decide t or t-2 or t-1 
                src_t = t_index - lag # if lag = 2, we need to get t_index-2 in p_7tb_cl
                if src_t < 0:
                    src_t = 0
                x_val = float(p_7tb_cl[base_dim, src_t, b_index]) # one entry of x_feat_f
                if bool(scale_io_by_20) and (base_dim == 0 or base_dim == 1):
                    x_val = x_val / 20.0
                x_feat_f[f_index] = x_val

            x_norm_f = (x_feat_f - mu_f) / std_f  # shape (F,)
            if x_cap is not None:
                x_norm_f = np.clip(x_norm_f, -x_cap, x_cap)  # shape (F,)

            x_tensor = torch.tensor(
                x_norm_f.reshape(1, 1, f_count), dtype=torch.float32, device=device
            )  # shape (1,1,F) since model() accepts (B,T,F) inputs. model () output (B,T,J)
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
            k_j = k_11j.reshape(j_count)  # shape (J,)
            k_jtb_cl[:, t_index, b_index] = k_j

            # s_j: (J,), branch FIR inputs at current time.
            s_j = k_j * u_now
            if m_count > 1:
                s_hist_jm[:, 1:] = s_hist_jm[:, 0:-1] # shift history by one step to discard old memory
            s_hist_jm[:, 0] = s_j   # s_hist_jm[1,:] = s_1(t) s_1(t-1) .... s_1(t-M+1)

            # g_arr * s_hist_jm is elementwise product; summing axis=1 gives branch convolution output.
            v_j = np.sum(g_arr * s_hist_jm, axis=1)  # shape (J,).
            y_branch_j = k_j * v_j  # shape (J,)
            y_hat_now = float(np.sum(y_branch_j))

            y_hat_tb_cl[t_index, b_index] = y_hat_now
            y_branch_jtb_cl[:, t_index, b_index] = y_branch_j

            # Update undelayed channels for next step (delay1 behavior on channels 1..6).
            if bool(fixed_uy_scale):
                y_scale_run = float(y_scale_fixed)  # scalar fixed y scale
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

    return y_hat_tb_cl, y_branch_jtb_cl, k_jtb_cl, p_7tb_cl


def train_step1(u_tb: np.ndarray, y_tb: np.ndarray, p_7tb: np.ndarray,
                cfg: dict[str, Any],
                initial_model_state_dict: dict[str,Any] | None = None) -> dict[str, Any]:
    """
      Train Step1 minimal NFIR model.

      Abbreivations:
      - T means num of samples per batch, 
      - B means number of batch 
      - F means dimension of extended scheduling signal (or called feature)
      - J = number of NFIR branches
      - M = FIR order

      Input dimensions: 
      - u_tb: numpy.ndarray, shape (T,B)
      - y_tb: numpy.ndarray, shape (T,B)
      - p_7tb: numpy.ndarray, shape (7,T,B)
      - cfg: configuration dict
      - initial_model_state_dict: Optional NN parameter dict.
                                  Same structure as result["model_state_dict"] 
                                  from a previous Step1 run.

      Main internal tensor dimensions:
      - p_ext_ftb: (F,T,B)
      - x_all_btf: (B,T,F)
      - u_all_bt: (B,T)
      - y_all_bt: (B,T)
      - g_jm: (J,M)

      Output dictionary includes:
      - y_hat_tb: (T,B)
      - y_branch_jtb: (J,T,B)
      - k_jtb: (J,T,B)
      - split indices and full history arrays
      - export metadata fields
    """
    # Validate NN configuration.
    _validate_config(cfg)
    
    # Set seeds for NN training.
    np.random.seed(int( cfg["random_seed"]  ))
    torch.manual_seed(  int(  cfg["random_seed"]   )   )

    # Convert training data to numeric arrays.
    u_tb = np.asarray(u_tb, dtype=float)
    y_tb = np.asarray(y_tb, dtype=float)
    p_7tb = np.asarray(p_7tb, dtype=float)

    # Check training data dimensions.
    if u_tb.ndim != 2 or y_tb.ndim != 2:
        raise ValueError("u_tb and y_tb must be 2D arrays with shape (T,B)")
    if p_7tb.shape != (7, u_tb.shape[0], u_tb.shape[1]):
        raise ValueError("p_7tb must match shape (7,T,B) aligned with u_tb")

    # Read num of batches for training
    n_batch = int( u_tb.shape[1] ) # u_tb: numpy.ndarray, shape (T,B)
    split_counts = cfg["train_val_test_split"]
    split_total = int(split_counts[0]) + int(split_counts[1]) + int(split_counts[2])
    if split_total != n_batch:
        raise ValueError(
            f"train_val_test_split must sum to n_batch={n_batch}, got {split_counts}"
        )

    # Split the batches to train, validation and test. 
    """
      Split the dataset into training, validation, and test sets. 
      The training set is used to train the model, 
      the validation set is used to tune hyperparameters 
        and monitor model performance, 
      and the test set is used to evaluate 
        the final model's performance on unseen data.
    """
    tr_idx, va_idx, te_idx = features.split_batch_indices(
        n_batch=n_batch,
        split_counts= cfg["train_val_test_split"],
        split_seed= int( cfg["split_seed"] ),
        shuffle= bool(cfg["shuffle_split"])
    )

    # Build extended scheduling signal/feature STRUCTURE/feature map from scheduling signal p_t
    # Dimension of feature structure/ feature map: (F,2)
    feature_map = features.build_feature_map(
        active_dims=cfg["active_dims"],
        delay_steps=cfg["delay_steps_by_dim"])

    # Given feature STRUCTRURE/feature map, we apply it to the data of p_t, 
    # to get the data of feature with shape (F,T,B).
    p_ext_ftb = features.build_p_ext_from_p7(
        p_7tb=p_7tb,
        feature_map=feature_map,
        scale_io_by_20=bool(cfg["scale_io_by_20"]),
    )
    # Then shape it to (B,T,F) so that NN can process easier
    x_all_btf = np.transpose(p_ext_ftb, (2,1,0)).astype(np.float32)

    # Reshape the io data to (B,T) and specify float point accuracy
    u_all_bt = np.transpose(u_tb, (1, 0)).astype(np.float32)
    y_all_bt = np.transpose(y_tb, (1, 0)).astype(np.float32)

    # Normalise the feature data. 
    mu_f, std_f = _compute_feature_norm_stats(x_all_btf[tr_idx], cfg["feature_norm_mode"])
    # Apply feature normalization to full dataset.
    x_all_btf = (x_all_btf - mu_f[None, None, :]) / std_f[None, None, :]
    # x_all_btf: shape (B,T,F), normalized feature data used by NN training.
    if cfg.get("x_max", None) is not None:
        x_max = float(cfg["x_max"])  # scalar clip bound for normalized feature vector x
        x_all_btf = np.clip(x_all_btf, -x_max, x_max)

    # fir_source_type_text: scalar string, FIR source type ("exponential" or "step2_pkl").
    fir_source_type_text = str(cfg["fir_source_type"]).strip().lower()
    # fir_source_path_used: scalar string path used for FIR import, else None.
    fir_source_path_used: str | None = None

    # Build FIR filter coefficients by selected source type.
    if fir_source_type_text == "exponential":
        g_bank_jm, taus_j, gains_j = features.make_fixed_exponential_fir_bank(
            n_branch=int(cfg["n_branch"]),
            m_fir=int(cfg["m_fir"]),
            dt=float(cfg["dt"]),
            seed=int(cfg["fir_seed"]),
            tau_min=float(cfg["fir_tau_min"]),
            tau_max=float(cfg["fir_tau_max"]),
            gain_min=float(cfg["fir_gain_min"]),
            gain_max=float(cfg["fir_gain_max"]),
        )
    else:
        # step2_pkl_path_text: scalar string path to Step2 pickle file.
        step2_pkl_path_text = str(cfg["fir_source_step2_pkl"])
        # g_bank_loaded_jm: np.ndarray, shape (J,M), FIR bank loaded from Step2 output.
        g_bank_loaded_jm = _load_fir_bank_from_step2_pkl(step2_pkl_path_text)

        # j_count_expected: scalar int, expected number of branches J from Step1 cfg.
        j_count_expected = int(cfg["n_branch"])
        # m_fir_expected: scalar int, expected FIR length M from Step1 cfg.
        m_fir_expected = int(cfg["m_fir"])
        # g_bank_expected_shape_jm: tuple(int,int), expected FIR matrix shape (J,M).
        g_bank_expected_shape_jm = (j_count_expected, m_fir_expected)
        # g_bank_loaded_shape_jm: tuple(int,int), loaded FIR matrix shape (J,M).
        g_bank_loaded_shape_jm = (int(g_bank_loaded_jm.shape[0]), int(g_bank_loaded_jm.shape[1]))

        if g_bank_loaded_shape_jm != g_bank_expected_shape_jm:
            raise ValueError(
                "Step2 g_bank shape mismatch: "
                f"expected {g_bank_expected_shape_jm}, got {g_bank_loaded_shape_jm}"
            )

        g_bank_jm = g_bank_loaded_jm
        # taus_j: np.ndarray, shape (J,), placeholder for imported FIR (tau not provided by Step2 here).
        taus_j = np.full((j_count_expected,), np.nan, dtype=float)
        # gains_j: np.ndarray, shape (J,), placeholder for imported FIR (gain not provided by Step2 here).
        gains_j = np.full((j_count_expected,), np.nan, dtype=float)
        fir_source_path_used = step2_pkl_path_text

    # Resolve runtime backend from cfg.
    # This is the only point where backend selection is decided.
    # All tensors/model created below use this same resolved device.
    device = _resolve_runtime_device(cfg["device"])

    # Convert numpy training data to torch tensors 
    x_t = torch.tensor(x_all_btf, dtype=torch.float32, device=device)
    u_t = torch.tensor(u_all_bt, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_all_bt, dtype=torch.float32, device=device)
    g_t = torch.tensor(g_bank_jm, dtype = torch.float32, device=device)

    # Init the NN structure
    model = SharedMLP(
        input_dim=int(x_all_btf.shape[-1]), # F 
        n_branch= int(cfg["n_branch"]), # B, the output dimension of NN
        hidden_dims=cfg["hidden_dims"],
        hidden_activation=str(cfg.get("mlp_hidden_activation", "tanh")),
        output_activation=str(cfg.get("mlp_output_activation", "tanh")),
    ).to(device)

    if initial_model_state_dict is not None:
        model.load_state_dict(initial_model_state_dict)

    # Init the NN training algorithm
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = float(cfg["learning_rate"]), # step size.
        weight_decay=float(cfg["weight_decay"]) # L2 regularisation weight
    )

    # Init NN training logging
    best_state = copy.deepcopy(model.state_dict()) # NN parameters
    best_val = np.inf # best  objective cost for valuation data
    best_epoch = -1 # which iteration/epoch gives the best results
    bad_count = 0 # increment when an epoch does not improve over the previous best value

    # Log  
    history_epoch = []
    history_train = [] # for train signals 
    history_val = [] # for validation signals
    history_test = [] # for test signals
  
    # Convert train indices to integer array.
    tr_idx_base = np.asarray(tr_idx, dtype=int)

    skip_open_loop_training = bool(cfg["skip_open_loop_training"])
    # if skip_open_loop_training and initial_model_state_dict is None:
    #     raise ValueError("skip_open_loop_training=True requires initial_model_state_dict.")
    if skip_open_loop_training:
        best_epoch = 0

    nn_open_loop_train_time_sec = 0.0  # scalar seconds
    nn_bptt_train_time_sec = 0.0       # scalar seconds
    
    if not skip_open_loop_training:
        # Use verbose option to decide whether to print out
        if bool(cfg["verbose"]):
            print("Printing training progress")
            print(
                f"[min-Step1] device={device} | split(train/val/test)="
                f"{len(tr_idx)}/{len(va_idx)}/{len(te_idx)}"
            )
            # Print explicit backend evidence so user can confirm GPU usage.
            device_info = _device_report(device)
            print(f"[min-Step1 GPU] device_backend={device_info['backend']} | device={device_info['device']}")
            for detail_line in device_info["details"]:
                print(f"[min-Step1] {detail_line}")
            print(
                f"[min-Step1] fir_source={fir_source_type_text} | epochs={cfg['max_epochs']}"
            )
        
        _sync_device_for_timing(device)
        timer1 = time.time()
        # Training loop: loop over epochs
        for epoch in range(1, int(cfg["max_epochs"]) + 1):
            # Set NN model to training mode
            model.train()
            
            # We have 16 batches of data for training and 
            # We need to break it to mini batches
            # So we need to make sure for each epoch, the content of 
            # mini batches are different. 

            epoch_perm = tr_idx_base.copy() # tr_idx_base = [2,5, 8, 10, 15] for example if we just have 5 batches of training data
            if bool(cfg["deterministic_epoch_shuffle"]):
                rng_epoch = np.random.default_rng(epoch + int( cfg["split_seed"] ) ) # choose a different seed for each epoch
                rng_epoch.shuffle(epoch_perm) # shuffle with the new seed
            # so epoch_perm now = [8, 5, 2, 15,10] for example 


            # Now we have 16 batches of train data shuffled, we need to break into mini-batches
            mini_batch_size = int( min(    int(cfg["batch_size"]), len(epoch_perm)  )) # how many batches per mini-batches

            # We train each mini-batch to update once the NN parameters
            for start in range(0, len(epoch_perm), mini_batch_size): # range(start, stop, stepsize)
                idx = epoch_perm[start:start + mini_batch_size] # e.g. idx = [8 5], then [2 15], then 10 if mini_batch_size = 2

                # slice data by batch index
                # x_t is  (B,T,F), x_t[0] returns a matrix of size (T,F), so first batch data
                # x_t[[0,1,2]] returns a tensor of size (3,T,F)
                xb = x_t[idx] 
                ub = u_t[idx]
                yb = y_t[idx]

                # clear the gradient computed in the previous mini-batch update
                optimizer.zero_grad(set_to_none = True)

                # Get NN symbolic output for current training data
                y_hat_b, _, _ = nfir_forward(model, xb, ub, g_t)

                # Compute the symbolic loss.
                loss = F.mse_loss(y_hat_b, yb) # import torch.nn.functional as F

                # Compute gradients with backpropagation.
                # grad is stored/updated inside model.parameters()
                loss.backward()

                # Clip gradients if they are too large.
                if cfg["grad_clip_norm"] is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip_norm"]))

                # Perform parameter update using the specified optimiser.
                optimizer.step()
            
            # Now all mini-batches have been trained, we record NN performance
            tr_loss = _eval_mse(model, x_t[tr_idx], u_t[tr_idx], y_t[tr_idx], g_t)
            va_loss = _eval_mse(model, x_t[va_idx], u_t[va_idx], y_t[va_idx], g_t)
            te_loss = _eval_mse(model, x_t[te_idx], u_t[te_idx], y_t[te_idx], g_t)


            # Append history values.
            history_epoch.append(epoch) # epoch number, e.g. the first or second 
            history_train.append(tr_loss)
            history_val.append(va_loss)
            history_test.append(te_loss)

            # Check if validation improves for this epoch 
            improved = va_loss < (best_val - float( cfg["early_stopping_min_delta"]  ))
            if improved:
                best_val = va_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                bad_count = 0 # reset bad_count
            else:
                bad_count = bad_count + 1

            # Optional logging for selected epochs.
            if bool(cfg["verbose"]):
                should_log = False
                if epoch == 1:
                    should_log = True
                if epoch % int(cfg["log_every"]) == 0:
                    should_log = True
                if epoch == int(cfg["max_epochs"]):
                    should_log = True
                if improved:
                    should_log = True

                if should_log:
                    pct = 100.0 * float(epoch) / float(cfg["max_epochs"])
                    print(
                        f"[min-Step1] epoch {epoch:4d}/{cfg['max_epochs']} ({pct:6.2f}%) "
                        f"| train={tr_loss:.6f} val={va_loss:.6f} test={te_loss:.6f} "
                        f"| best_val={best_val:.6f} @ {best_epoch}"
                    )
            
            # Early stopping check 
            if bool(cfg["early_stopping"]):
                if bad_count >= int(cfg["early_stopping_patience"]):
                    if bool(cfg["verbose"]):
                        print(
                            f"[min-Step1] eary stopping at epoch {epoch}; "
                            f"patience = {cfg['early_stopping_patience']}"
                        )
                    break # exit epoch training loop
        
        _sync_device_for_timing(device)
        nn_open_loop_train_time_sec = time.time() - timer1
        print('In step1 open loop training. Time to train = ', nn_open_loop_train_time_sec)
    else:
        best_val = _eval_mse(model, x_t[va_idx], u_t[va_idx], y_t[va_idx], g_t)
        if bool(cfg["verbose"]):
            print("[min-Step1] skip_open_loop_training=True; using initial model state")

    # Restore best model state
    model.load_state_dict(best_state)
    """ BPTT block starts here 
    """
    _sync_device_for_timing(device)
    timer2 = time.time()

    bptt_history_epoch = []
    bptt_history_train_loss = []
    bptt_history_val_loss = []
    bptt_history_test_loss = []
    bptt_history_lr = []
    bptt_best_epoch = -1 
    bptt_best_val_loss = float("inf")
    bptt_no_improve = 0
    bptt_best_state = copy.deepcopy(model.state_dict())
    mu_f_torch = torch.tensor(mu_f, dtype=torch.float32, device=device)
    std_f_torch = torch.tensor(std_f, dtype=torch.float32, device=device)
    u_bt_torch = u_t # rename to make the bt clear; shape (B,T)
    y_bt_torch = y_t # rename to make the bt clear; shape (B,T)
    g_jm_torch = g_t # shape (J,M)

    if bool(cfg["bptt_finetune_enable"]) and (int(cfg["bptt_max_epochs"]) > 0):
        # This guarantees old behavior is unchanged unless the user explicitly enables BPTT.
        
        bptt_optimiser = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg["bptt_learning_rate"]),
            weight_decay= float(cfg["weight_decay"]))
        
        """ For controlling the step size of bptt
        """
        bptt_scheduler_name = str(cfg["bptt_lr_scheduler"])
        bptt_scheduler = None
        bptt_steps_per_epoch = int(np.ceil(len(tr_idx) / int(cfg["bptt_batch_size"])))  # scalar, updates per BPTT epoch. 
                                    # len(tr_idx) = how many batches in one epoch
                                    # bptt_batch_size = how many batches in each mini batches 
                                    # so divide is how many mini batches per epoch 
                                    # The solver update once every mino batches. 
                                    # So this is how many solver updates per epoch 
        if bptt_scheduler_name == "plateau":
            bptt_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                bptt_optimiser,
                mode="min",
                factor=float(cfg["bptt_lr_decay_factor"]), # Factor by which the learning rate will be reduced. 
                                                            # new_lr = lr * factor. Default: 0.1.
                patience=int(cfg["bptt_lr_decay_patience"]), # patience : The number of allowed epochs with no improvement after which the learning rate will be reduced.
                min_lr=float(cfg["bptt_min_learning_rate"]), # minimal learning rate
            )
        elif bptt_scheduler_name == "exponential":
            bptt_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                bptt_optimiser,
                gamma=float(cfg["bptt_lr_gamma"]),
            )
        elif bptt_scheduler_name == "cosine":
            bptt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                bptt_optimiser,
                T_max=int(cfg["bptt_max_epochs"]),
                eta_min=float(cfg["bptt_min_learning_rate"]),
            )
        elif bptt_scheduler_name == "warmup_cosine":
            warmup_epochs = int(cfg["bptt_warmup_epochs"])
            cosine_epochs = max(1, int(cfg["bptt_max_epochs"]) - warmup_epochs)
            bptt_scheduler = torch.optim.lr_scheduler.SequentialLR(
                bptt_optimiser,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(
                        bptt_optimiser,
                        start_factor=0.1,
                        total_iters=max(1, warmup_epochs),
                    ),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        bptt_optimiser,
                        T_max=cosine_epochs,
                        eta_min=float(cfg["bptt_min_learning_rate"]),
                    ),
                ],
                milestones=[warmup_epochs],
            )
        elif bptt_scheduler_name == "onecycle":
            bptt_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                bptt_optimiser,
                max_lr=float(cfg["bptt_learning_rate"]),
                epochs=int(cfg["bptt_max_epochs"]),
                steps_per_epoch=bptt_steps_per_epoch,
                pct_start=float(cfg["bptt_onecycle_pct_start"]),
                anneal_strategy="cos",
                div_factor=float(cfg["bptt_onecycle_div_factor"]),
                final_div_factor=float(cfg["bptt_onecycle_final_div_factor"]),
                cycle_momentum=False,
            )
        
        print("[min-Step1][BPTT] Starting closed-loop fine-tuning")
        # Epoch 0 records closed-loop performance before any BPTT update.
        with torch.no_grad():
            model.eval()
            y_hat_tra_bt_cl_torch, _, _ = bptt_closed_loop_rollout(
                model=model,
                u_bt=u_bt_torch[tr_idx],
                y_bt=y_bt_torch[tr_idx],
                g_jm=g_jm_torch,
                feature_map=feature_map,
                feature_mean_f=mu_f_torch,
                feature_std_f=std_f_torch,
                cfg=cfg
            )
            y_hat_val_bt_cl_torch, _, _ = bptt_closed_loop_rollout(
                model=model,
                u_bt=u_bt_torch[va_idx],
                y_bt=y_bt_torch[va_idx],
                g_jm=g_jm_torch,
                feature_map=feature_map,
                feature_mean_f=mu_f_torch,
                feature_std_f=std_f_torch,
                cfg=cfg
            )
            y_hat_tes_bt_cl_torch, _, _ = bptt_closed_loop_rollout(
                model=model,
                u_bt=u_bt_torch[te_idx],
                y_bt=y_bt_torch[te_idx],
                g_jm=g_jm_torch,
                feature_map=feature_map,
                feature_mean_f=mu_f_torch,
                feature_std_f=std_f_torch,
                cfg=cfg
            )
            bptt_tr_loss = F.mse_loss(y_hat_tra_bt_cl_torch, y_bt_torch[tr_idx]).item()
            bptt_va_loss = F.mse_loss(y_hat_val_bt_cl_torch, y_bt_torch[va_idx]).item()
            bptt_te_loss = F.mse_loss(y_hat_tes_bt_cl_torch, y_bt_torch[te_idx]).item()

        bptt_history_epoch.append(0)
        bptt_history_train_loss.append(bptt_tr_loss)
        bptt_history_val_loss.append(bptt_va_loss)
        bptt_history_test_loss.append(bptt_te_loss)
        bptt_current_lr = float(bptt_optimiser.param_groups[0]["lr"])
        bptt_history_lr.append(bptt_current_lr)
        bptt_best_epoch = 0
        bptt_best_val_loss = bptt_va_loss
        bptt_best_state = copy.deepcopy(model.state_dict())

        if bool(cfg["verbose"]):
            print(
                f"[min-Step1-BPTT] epoch    0/{cfg['bptt_max_epochs']} (  0.00%) "
                f"| train={bptt_tr_loss:.6f} val={bptt_va_loss:.6f} test={bptt_te_loss:.6f} "
                f"| best_val={bptt_best_val_loss:.6f} @ {bptt_best_epoch}"
                f"| lr={bptt_current_lr:.3e}"
            )

        for epoch in range(1, int(cfg["bptt_max_epochs"]) + 1):
            model.train()

            """ Note
                We have 16 batches of data for training and 
                # We need to break it to mini batches
                # So we need to make sure for each epoch, the content of 
                # mini batches are different. 
            """
            # Build BPTT mini-batches from train trajectory indices for this BPTT epoch.
            bptt_epoch_perm = np.asarray(tr_idx, dtype=int).copy() # shape (B_train,)
            if bool(cfg["deterministic_epoch_shuffle"]):
                rng_bptt_epoch = np.random.default_rng(
                    int(cfg["split_seed"]) + 100000 + epoch
                )
                rng_bptt_epoch.shuffle(bptt_epoch_perm)

            # Now we have train batches shuffled, we need to break into mini-batches.
            mini_batch_size = int(min(int(cfg["bptt_batch_size"]), len(bptt_epoch_perm))) # how many batches per mini-batches

            # We train each mini-batch to update once the NN parameters
            for start in range(0, len(bptt_epoch_perm), mini_batch_size): # range(start, stop, stepsize)
                idx = bptt_epoch_perm[start:start + mini_batch_size] # e.g. idx = [8, 5], then [2, 15], then 10 if mini_batch_size = 2

                """ Python trick
                    >>> idx
                    [0, 2]
                    >>> g_bt[idx]
                    tensor([[0.2134, 0.2497, 0.0475, 0.6955, 0.2507, 0.4406, 0.5078, 0.4731, 0.0733,
                            0.5066],
                            [0.0863, 0.2110, 0.4360, 0.3937, 0.2805, 0.6881, 0.8768, 0.3364, 0.9020,
                            0.3712]])
                    >>> g_bt[idx].shape
                    torch.Size([2, 10])
                    >>> g_bt.shape
                    torch.Size([5, 10])
                    >>> g_bt
                    tensor([[0.2134, 0.2497, 0.0475, 0.6955, 0.2507, 0.4406, 0.5078, 0.4731, 0.0733,
                            0.5066],
                            [0.3377, 0.9691, 0.2746, 0.9508, 0.7801, 0.8540, 0.1368, 0.1203, 0.2442,
                            0.9520],
                            [0.0863, 0.2110, 0.4360, 0.3937, 0.2805, 0.6881, 0.8768, 0.3364, 0.9020,
                            0.3712],
                            [0.7878, 0.7859, 0.1159, 0.6627, 0.7333, 0.5350, 0.7595, 0.4391, 0.2475,
                            0.3789],
                            [0.4059, 0.0955, 0.7978, 0.4156, 0.6017, 0.9433, 0.1399, 0.0979, 0.2353,
                            0.1006]])
                """
                # slice data by batch index
                u_minibt_torch = u_bt_torch[idx] # u_minibt: shape (mini_batch_size,T)
                y_minibt_torch = y_bt_torch[idx]

                # clear the gradient computed in the previous mini-batch update
                bptt_optimiser.zero_grad(set_to_none = True)

                # Get NN symbolic output for current training data
                # y_hat_b, _, _ = nfir_forward(model, xb, ub, g_t)
                y_hat_minibt_cl_torch, _, _ = bptt_closed_loop_rollout(
                    model=model,
                    u_bt=u_minibt_torch,
                    y_bt=y_minibt_torch,
                    g_jm=g_jm_torch,
                    feature_map=feature_map,
                    feature_mean_f=mu_f_torch,
                    feature_std_f=std_f_torch,
                    cfg=cfg
                )

                # Compute the symbolic loss.
                loss = F.mse_loss(y_hat_minibt_cl_torch, y_minibt_torch) # import torch.nn.functional as F

                # Compute gradients with backpropagation.
                # grad is stored/updated inside model.parameters()
                loss.backward()

                # Clip gradients if they are too large.
                if cfg["bptt_grad_clip_norm"] is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), float(cfg["bptt_grad_clip_norm"]))

                # Perform parameter update using the specified optimiser.
                bptt_optimiser.step()

                
                """ OneCycleLR must step after every optimizer update.
                    OneCycleLR is update-based, not epoch-based.
                    With 300 train batches and bptt_batch_size = 4:
                    bptt_steps_per_epoch = 75
                    So OneCycleLR expects 75 scheduler steps per epoch.
                """
                if bptt_scheduler_name == "onecycle":
                    bptt_scheduler.step()


            # Now all mini-batches FOR THIS epoch have been trained, we record NN performance
            with torch.no_grad():
                model.eval()
                y_hat_tra_bt_cl_torch, _, _ = bptt_closed_loop_rollout(
                    model=model,
                    u_bt=u_bt_torch[tr_idx],
                    y_bt=y_bt_torch[tr_idx],
                    g_jm=g_jm_torch,
                    feature_map=feature_map,
                    feature_mean_f=mu_f_torch,
                    feature_std_f=std_f_torch,
                    cfg=cfg
                )
                y_hat_val_bt_cl_torch, _, _ = bptt_closed_loop_rollout(
                    model=model,
                    u_bt=u_bt_torch[va_idx],
                    y_bt=y_bt_torch[va_idx],
                    g_jm=g_jm_torch,
                    feature_map=feature_map,
                    feature_mean_f=mu_f_torch,
                    feature_std_f=std_f_torch,
                    cfg=cfg
                )
                y_hat_tes_bt_cl_torch, _, _ = bptt_closed_loop_rollout(
                    model=model,
                    u_bt=u_bt_torch[te_idx],
                    y_bt=y_bt_torch[te_idx],
                    g_jm=g_jm_torch,
                    feature_map=feature_map,
                    feature_mean_f=mu_f_torch,
                    feature_std_f=std_f_torch,
                    cfg=cfg
                )
                # Use .item() to return scalar Python values.
                bptt_tr_loss = F.mse_loss(y_hat_tra_bt_cl_torch, y_bt_torch[tr_idx]).item() # import torch.nn.functional as F
                bptt_va_loss = F.mse_loss(y_hat_val_bt_cl_torch, y_bt_torch[va_idx]).item() # import torch.nn.functional as F
                bptt_te_loss = F.mse_loss(y_hat_tes_bt_cl_torch, y_bt_torch[te_idx]).item() # import torch.nn.functional as F
        

            if bptt_scheduler is not None and bptt_scheduler_name == "plateau":
                bptt_scheduler.step(bptt_va_loss)
            elif bptt_scheduler is not None and bptt_scheduler_name != "onecycle":
                bptt_scheduler.step()

            bptt_current_lr = float(bptt_optimiser.param_groups[0]["lr"])

            # Append history values.
            bptt_history_epoch.append(epoch) # epoch number, e.g. the first or second 
            bptt_history_train_loss.append(bptt_tr_loss)
            bptt_history_val_loss.append(bptt_va_loss)
            bptt_history_test_loss.append(bptt_te_loss)
            bptt_history_lr.append(bptt_current_lr)

            # Check if validation improves for this epoch 
            improved = bptt_va_loss < (bptt_best_val_loss - float( cfg["early_stopping_min_delta"]  ))
            if improved:
                bptt_best_val_loss = bptt_va_loss
                bptt_best_epoch = epoch
                bptt_best_state = copy.deepcopy(model.state_dict())
                bptt_no_improve = 0 # reset bad_count
            else:
                bptt_no_improve = bptt_no_improve + 1

            # Optional logging for selected epochs.
            if bool(cfg["verbose"]):
                should_log = False
                if epoch == 1:
                    should_log = True
                if epoch % int(cfg["log_every"]) == 0:
                    should_log = True
                if epoch == int(cfg["bptt_max_epochs"]):
                    should_log = True
                if improved:
                    should_log = True

                if should_log:
                    pct = 100.0 * float(epoch) / float(cfg["bptt_max_epochs"])
                    print(
                        f"[min-Step1-BPTT] epoch {epoch:4d}/{cfg['bptt_max_epochs']} ({pct:6.2f}%) "
                        f"| train={bptt_tr_loss:.6f} val={bptt_va_loss:.6f} test={bptt_te_loss:.6f} "
                        f"| best_val={bptt_best_val_loss:.6f} @ {bptt_best_epoch}"
                        f"| lr={bptt_current_lr:.3e}"
                    )

            # Early stopping check 
            if bool(cfg["bptt_early_stopping"]):
                if bptt_no_improve >= int(cfg["bptt_early_stopping_patience"]):
                    if bool(cfg["verbose"]):
                        print(
                            f"[min-Step1-BPTT] early stopping at epoch {epoch}; "
                            f"patience = {cfg['bptt_early_stopping_patience']}"
                        )
                    break # exit epoch training loop
    
        # All epoch training finished. Restore best model state.
        model.load_state_dict(bptt_best_state)

    """ BPTT block ends here 
    """
    _sync_device_for_timing(device)
    nn_bptt_train_time_sec = time.time() - timer2
    print('In step1 BPTT loop training. Time to BPT = ', nn_bptt_train_time_sec)

    # Inference over all traj
    #  - y_hat_bt: numpy.ndarray, shape (B,T)
    #- y_branch_btj: numpy.ndarray, shape (B,T,J), J = number of NFIR branches
    # - k_btj: numpy.ndarray, shape (B,T,J)
    y_hat_bt, y_branch_btj, k_btj = _infer_all(model, x_t, u_t, g_t)

    # Reshape 
    y_hat_tb = np.transpose(y_hat_bt, (1,0))
    y_branch_jtb = np.transpose(y_branch_btj, (2,1,0))
    k_jtb = np.transpose(k_btj, (2,1,0))

    # Closed-loop evaluation with causal p_7tb rebuild and running y-scale.
    y_hat_tb_cl, y_branch_jtb_cl, k_jtb_cl, p_7tb_cl = _infer_all_closed_loop(
        model=model,
        u_tb=np.asarray(u_tb, dtype=float),
        g_jm=np.asarray(g_bank_jm, dtype=float),
        feature_map=np.asarray(feature_map, dtype=int),
        feature_mean=np.asarray(mu_f, dtype=float),
        feature_std=np.asarray(std_f, dtype=float),
        scale_io_by_20=bool(cfg["scale_io_by_20"]),
        dt_sec=float(cfg["dt"]),
        fixed_uy_scale=bool(cfg["fixed_uy_scale"]),
        u_scale_fixed=float(cfg["u_scale_fixed"]),
        y_scale_fixed=float(cfg["y_scale_fixed"]),
        uy_scale_method=str(cfg.get("uy_scale_method", "divide")),
        u_max_after_scale=float(cfg.get("u_max_after_scale", 1.0)),
        y_max_after_scale=float(cfg.get("y_max_after_scale", 1.0)),
        x_max=cfg.get("x_max", None),
        device=device,
    )

    # Split arrays for open-loop and closed-loop summaries.
    y_train_tb = np.asarray(y_tb, dtype=float)[:, tr_idx]
    y_val_tb = np.asarray(y_tb, dtype=float)[:, va_idx]
    y_test_tb = np.asarray(y_tb, dtype=float)[:, te_idx]

    y_pre_train_tb = np.asarray(y_hat_tb, dtype=float)[:, tr_idx]
    y_pre_val_tb = np.asarray(y_hat_tb, dtype=float)[:, va_idx]
    y_pre_test_tb = np.asarray(y_hat_tb, dtype=float)[:, te_idx]

    y_pre_train_tb_cl = np.asarray(y_hat_tb_cl, dtype=float)[:, tr_idx]
    y_pre_val_tb_cl = np.asarray(y_hat_tb_cl, dtype=float)[:, va_idx]
    y_pre_test_tb_cl = np.asarray(y_hat_tb_cl, dtype=float)[:, te_idx]

    train_mse_ol = float(np.mean((y_pre_train_tb - y_train_tb) ** 2))
    val_mse_ol = float(np.mean((y_pre_val_tb - y_val_tb) ** 2))
    test_mse_ol = float(np.mean((y_pre_test_tb - y_test_tb) ** 2))

    train_mse_cl = float(np.mean((y_pre_train_tb_cl - y_train_tb) ** 2))
    val_mse_cl = float(np.mean((y_pre_val_tb_cl - y_val_tb) ** 2))
    test_mse_cl = float(np.mean((y_pre_test_tb_cl - y_test_tb) ** 2))

    # Build results that can be exported easily
    result = {}
    result["schema_version"] = str(cfg["schema_version"])
    result["timestamp_utc"] = io_data.utc_now_iso()
    result["mode"] = str(cfg["mode"])
    result["run_name"] = str(cfg["run_name"])
    result["cfg"] = dict(cfg)
    result["device"] = str(device)
    result["model"] = model
    result["model_state_dict"] = copy.deepcopy(model.state_dict()) # NN para
    result["feature_map"] = np.asarray(feature_map, dtype=int)
    result["feature_mean"] = np.asarray(mu_f, dtype=float) # mean of feature data 
    result["feature_std"] = np.asarray(std_f, dtype=float) # std of feature data 
    result["x_max"] = cfg.get("x_max", None)
    result["g_bank"] = np.asarray(g_bank_jm, dtype=float)
    result["fir_taus"] = np.asarray(taus_j, dtype=float)
    result["fir_gains"] = np.asarray(gains_j, dtype=float)
    result["fir_source_type"] = str(fir_source_type_text)
    result["fir_source_path"] = fir_source_path_used
    result["fixed_uy_scale"] = bool(cfg["fixed_uy_scale"])
    result["u_scale_used_cl"] = float(cfg["u_scale_fixed"]) if bool(cfg["fixed_uy_scale"]) else float("nan")
    result["y_scale_used_cl"] = float(cfg["y_scale_fixed"]) if bool(cfg["fixed_uy_scale"]) else float("nan")
    result["split_source"] = "fresh_from_cfg"
    result["split_train_idx"] = np.asarray(tr_idx, dtype=int)
    result["split_val_idx"] = np.asarray(va_idx, dtype=int)
    result["split_test_idx"] = np.asarray(te_idx, dtype=int)
    result["history_epoch"] = np.asarray(history_epoch, dtype=int)
    result["history_train_loss"] = np.asarray(history_train, dtype=float)
    result["history_val_loss"] = np.asarray(history_val, dtype=float)
    result["history_test_loss"] = np.asarray(history_test, dtype=float)
    result["best_epoch"] = int(best_epoch)
    result["best_val_loss"] = float(best_val)
    result["y_hat_tb"] = np.asarray(y_hat_tb, dtype=float)
    result["y_hat_tb_cl"] = np.asarray(y_hat_tb_cl, dtype=float)
    result["y_branch_jtb"] = np.asarray(y_branch_jtb, dtype=float)
    result["y_branch_jtb_cl"] = np.asarray(y_branch_jtb_cl, dtype=float)
    result["k_jtb"] = np.asarray(k_jtb, dtype=float)
    # result["k_jtb_cl"] = np.asarray(k_jtb_cl, dtype=float)
    result["u_tb"] = np.asarray(u_tb, dtype=float)
    result["y_tb"] = np.asarray(y_tb, dtype=float)
    result["p_7tb"] = np.asarray(p_7tb, dtype=float)
    result["p_7tb_cl"] = np.asarray(p_7tb_cl, dtype=float)
    result["y_pre_train_batch"] = np.asarray(y_pre_train_tb, dtype=float)
    result["y_pre_val_batch"] = np.asarray(y_pre_val_tb, dtype=float)
    result["y_pre_test_batch"] = np.asarray(y_pre_test_tb, dtype=float)
    result["y_pre_train_batch_cl"] = np.asarray(y_pre_train_tb_cl, dtype=float)
    result["y_pre_val_batch_cl"] = np.asarray(y_pre_val_tb_cl, dtype=float)
    result["y_pre_test_batch_cl"] = np.asarray(y_pre_test_tb_cl, dtype=float)
    result["y_train_batch"] = np.asarray(y_train_tb, dtype=float)
    result["y_val_batch"] = np.asarray(y_val_tb, dtype=float)
    result["y_test_batch"] = np.asarray(y_test_tb, dtype=float)
    result["train_mse_ol"] = train_mse_ol
    result["val_mse_ol"] = val_mse_ol
    result["test_mse_ol"] = test_mse_ol
    result["train_mse_cl"] = train_mse_cl
    result["val_mse_cl"] = val_mse_cl
    result["test_mse_cl"] = test_mse_cl
    result["lineage_prev_step1_path"] = None
    result["lineage_prev_step2_path"] = None
    result["lineage_freeze_split_path"] = None
    result["skip_open_loop_training"] = bool(cfg["skip_open_loop_training"])
    result["used_initial_model_state_dict"] = bool(initial_model_state_dict is not None)

    result["bptt_history_epoch"] = np.asarray(bptt_history_epoch, dtype=int)
    result["bptt_history_train_loss"] = np.asarray(bptt_history_train_loss, dtype=float)
    result["bptt_history_val_loss"]= np.asarray(bptt_history_val_loss, dtype=float)
    result["bptt_history_test_loss"]= np.asarray(bptt_history_test_loss, dtype=float)
    result["bptt_best_epoch"] =  int(bptt_best_epoch)
    result["bptt_best_val_loss"] =  float(bptt_best_val_loss)

    # bptt_history_epoch: shape (N+1,) bptt_history_lr:    shape (N+1,)
    result["bptt_history_lr"] = np.asarray(bptt_history_lr, dtype=float)
    result["bptt_lr_scheduler"] = str(cfg["bptt_lr_scheduler"])

    result["nn_open_loop_train_time_sec"] = float(nn_open_loop_train_time_sec)
    result["nn_bptt_train_time_sec"] = float(nn_bptt_train_time_sec)
    result["nn_train_time_sec"] = float(nn_open_loop_train_time_sec + nn_bptt_train_time_sec)

    if cfg["save_full_diagnostics"] == False:
        result["y_branch_jtb"] = np.zeros(1)
        result["y_branch_jtb_cl"] = np.zeros(1)
    return result


def run_from_mat_file(mat_path: str, out_dir: str, run_name: str,
                      cfg: dict[str,Any] | None = None,
                      initial_model_state_dict:dict[str,Any] | None = None) -> dict[str,Any]:
    """
      Convenience function: load MAT, train, export, return result.

      Input:
      - mat_path: path to MAT training data
      - out_dir: output folder path
      - run_name: output base name
      - cfg: optional config dict; if None, default config is used

      Output:
      - result dictionary from `train_step1`
        with extra keys:
        - "pkl_path": str
        - "mat_path": str
      
      Python syntex facts:
      - cfg: dict[str,Any] | None = None
        This means that the parameter cfg can either be a dictionary with string keys and values of any type, or it can be None. 
        The default value for cfg is None if it is not provided when the function is called.
    """

    if cfg is None:
        cfg = build_default_config()
    
    cfg_local = dict(cfg) # shallow copy to avoid mutating input dict
    cfg_local["run_name"] = str(run_name) # ensure run_name is string

    # Load training arrays from MAT file.
    loaded = io_data.load_training_mat(mat_path)
    u_tb = loaded["u_tb"]
    y_tb = loaded["y_tb"]
    p_7tb = loaded["p_7tb"]
    if bool(cfg_local.get("rebuild_p7_from_uy", False)):
        p_7tb = io_data.build_p7_from_u_y(
            u_tb=u_tb,
            y_tb=y_tb,
            ts=float(cfg_local["dt"]),
            fixed_uy_scale=bool(cfg_local.get("fixed_uy_scale", False)),
            u_scale_fixed=float(cfg_local.get("u_scale_fixed", 22.0)),
            y_scale_fixed=float(cfg_local.get("y_scale_fixed", 31.471743603975618)),
            uy_scale_method=str(cfg_local.get("uy_scale_method", "divide")),
            u_max_after_scale=float(cfg_local.get("u_max_after_scale", 1.0)),
            y_max_after_scale=float(cfg_local.get("y_max_after_scale", 1.0)),
        )

    # Train model.
    result = train_step1(u_tb=u_tb, y_tb=y_tb, p_7tb=p_7tb, cfg=cfg_local,
                         initial_model_state_dict=initial_model_state_dict)

    # Export results.
    pkl_path_, mat_path_ = io_data.save_step1_outputs(result, out_dir=out_dir, run_name=run_name)

    # Store output paths 
    # Store output paths in result for convenience.
    result["pkl_path"] = str(pkl_path_)
    result["mat_path"] = str(mat_path_)

    # Return results. 
    
    return result
