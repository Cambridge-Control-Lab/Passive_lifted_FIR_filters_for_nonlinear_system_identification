"""Exp1 NFIR alternating-training orchestration script.

Role in the workflow:
- This script runs the Exp1 outer loop for Algorithm 1 of
  arXiv:2508.05279v2.
- Each outer iteration first runs theta_G, the FIR/filter-parameter
  optimization step, and then theta_N, the neural-lifting training step.
- Legacy code variable names still contain ``step2`` for theta_G and
  ``step1`` for theta_N. Those identifiers are intentionally not renamed,
  because run names, saved files, and downstream scripts depend on them.

Data/result handoff:
- theta_G writes a ``*.pkl`` and ``*_train.mat`` result containing the solved
  FIR bank g_jm, shape (J,M).
- theta_N imports the latest theta_G ``*.pkl`` result, trains the lifting MLP,
  and saves model state plus predictions for the next theta_G iteration.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import pickle

EXP1_DIR = Path(__file__).resolve().parents[1]  # scalar Path: Exp1 folder.
ROOT = EXP1_DIR.parent  # scalar Path: repository root.
DATA_DIR = EXP1_DIR / "Training_data"  # scalar Path: training data folder.
out_dir = EXP1_DIR / "Results"  # scalar Path: output results folder.
root_path = str(ROOT)  # scalar string: repository root path for direct script runs.
if root_path not in sys.path:
    sys.path.insert(0, root_path)
print("ROOT is ", ROOT)
mat_path = DATA_DIR / "Data_M_NLdamper_500B_OneCart.mat"  # Set input MAT data path. Training data. 
out_dir.mkdir(parents=True, exist_ok=True)

from Exp1.Core_code import theta_N_core as train_core
from Exp1.Core_code import theta_G_core as step2_min_core
import numpy as np


"""Configuration note:
    When Step1 is run without initial_model_state_dict, the NN initialization
    is deterministic for the same random seed.
"""

dt = 0.02 
hidden_dims = (4, 4)  # shape (2,)
active_dims = (0,1,6)  # shape (n_active_dim,)
delay_steps_by_dim = {}  # keys are dim ids, values are delay steps.
max_iteration = 4
fixed_uy_scale = True
rebuild_p7_from_uy = True
u_scale_fixed = float(16.5)
y_scale_fixed = float(2.94)
uy_scale_method = "divide"
u_max_after_scale =  100000.0 
y_max_after_scale =  100000.0 
feature_norm_mode = "zscore"
m_fir = 100
n_branch_all_cases = int(20) # later we can change to 56 
is_passive = True
step2_l2reg = 1.0
save_full_diagnostics_ON = False
NN_para_inher_double_iter = True # If True, NN initial parameters come from the previous iteration.
"""Configuration note:
    When Step1 is run without initial_model_state_dict, the NN initialization
    is deterministic for the same random seed.
"""
parser = argparse.ArgumentParser()
parser.add_argument("--exp-id", default="e00")
parser.add_argument("--hidden-dims", default="4,4")
parser.add_argument("--active-dims", default="0,1,6")
parser.add_argument("--delay-steps-by-dim", default="{}")
parser.add_argument("--m-fir", type=int, default=m_fir)
parser.add_argument("--n-branch-all-cases", type=int, default=n_branch_all_cases)
parser.add_argument("--mat-path", default=None)
parser.add_argument("--noise-on", type=int, choices=[0, 1], default=0)
parser.add_argument("--is-passive", type=int, choices=[0, 1], default=1)
parser.add_argument("--max-iteration", type=int, default=max_iteration)
parser.add_argument("--step2-l2reg", type=float, default=step2_l2reg)
parser.add_argument("--last-bptt-on", type=int, choices=[0, 1], default=0)
args = parser.parse_args()

exp_id = args.exp_id
hidden_dims = tuple(int(s) for s in args.hidden_dims.split(","))  # shape (2,)
active_dims = tuple(int(s) for s in args.active_dims.split(","))  # shape (n_active_dim,)
delay_steps_by_dim = json.loads(args.delay_steps_by_dim)
delay_steps_by_dim = {int(k): int(v) for k, v in delay_steps_by_dim.items()}
m_fir = int(args.m_fir)
n_branch_all_cases = int(args.n_branch_all_cases)
noise_on = bool(args.noise_on)
if args.mat_path is not None:
    mat_path = Path(args.mat_path)
elif noise_on:
    mat_path = DATA_DIR / "Data_M_NLdamper_500B_OneCart_SNR10.mat"
else:
    mat_path = DATA_DIR / "Data_M_NLdamper_500B_OneCart.mat"
is_passive = bool(args.is_passive)
max_iteration = int(args.max_iteration)
step2_l2reg = float(args.step2_l2reg)
last_bptt_on = bool(args.last_bptt_on)

if not mat_path.is_file():
    raise FileNotFoundError(f"Missing MAT data file: {mat_path}")

active_dim_tag = "".join(str(i_dim) for i_dim in active_dims)
noise_tag = "noise" if noise_on else "nonoise"
passive_tag = "pas" if is_passive else "nopas"
bptt_tag = "lastbptt" if last_bptt_on else "nobptt"
l2_tag = f"l2{step2_l2reg:g}".replace(".", "p")
case_tag = f"run_NFIR_{exp_id}_h{hidden_dims[0]}_ad{active_dim_tag}_m{m_fir}_b{n_branch_all_cases}_{noise_tag}_{passive_tag}_ite{max_iteration}_{bptt_tag}_{l2_tag}_neoit"

step2_run_name_arr = [f"{case_tag}_s{20 + iter_idx}" for iter_idx in range(max_iteration)]
mode_step2_arr = ["random_nn"] + ["imported_nn"] * (max_iteration - 1)
step1_run_name_arr = [f"{case_tag}_s{10 + iter_idx}" for iter_idx in range(max_iteration)]
step1_run_name_pkl_arr = [f"{sname}.pkl" for sname in step1_run_name_arr]

# if len(step1_run_name_arr) != max_iteration or len(step2_run_name_arr) != max_iteration or len(mode_step2_arr) != max_iteration:
#     raise ValueError("legacy theta_N/theta_G run_name_arr size wrong")


for iter_idx in range(0,max_iteration):
    print("[Double-iteration] Starting")
    print(f"Current iteration = {iter_idx} / {max_iteration}")

    
    mode = mode_step2_arr[iter_idx]
    cfg_step2 = step2_min_core.build_default_config_min(mode)
    cfg_step2["n_branch"] = n_branch_all_cases
    cfg_step2["m_fir"] = m_fir
    cfg_step2["l2reg"] = step2_l2reg
    cfg_step2["save_full_diagnostics"] = save_full_diagnostics_ON # whether save all details 
    cfg_step2["k_source_mode"] = mode
    cfg_step2["run_name"] = step2_run_name_arr[iter_idx] # the file name of final output data after optimisation
    cfg_step2["out_dir"] = str(out_dir)  # which folder in this workspace to store the output results
    cfg_step2["source_data_mat"] = str(mat_path) #matlab file for training data
    
    cfg_step2["active_dims"] = active_dims # if just one dimension, then use (1,) or (2,) or (0,)
    cfg_step2["delay_steps_by_dim"] = delay_steps_by_dim
    cfg_step2["n_refine_iter"] = 0
    cfg_step2["iter_yhat_add_noise"] = False 
    cfg_step2["train_val_test_split"] = (70, 330, 100) 
    cfg_step2["imported_split_source"] = "cfg"   # "step1": use theta_N saved split; "cfg": use theta_G train_val_test_split.
    
    if mode == "poly_lifting":
        cfg_step2["x_max"] = 1.0
    else:
        cfg_step2["x_max"] = None  # imported_nn must match theta_N x_max.

    cfg_step2["ms_passivity"] = 10000 
    cfg_step2["feature_norm_mode"] = feature_norm_mode  # 'none' or 'zscore'
    cfg_step2["poly_basis_type"] = "legendre" # "monomial" is current x^n basis; "legendre" uses bounded orthogonal polynomials.
    cfg_step2["fixed_uy_scale"] = fixed_uy_scale
    cfg_step2["rebuild_p7_from_uy"] = rebuild_p7_from_uy 
    cfg_step2["u_scale_fixed"] = u_scale_fixed
    cfg_step2["y_scale_fixed"] =y_scale_fixed
    cfg_step2["uy_scale_method"] = uy_scale_method  # "softsign" => x/sqrt(x^2+s^2), "divide" => x/s
    cfg_step2["u_max_after_scale"] = u_max_after_scale # clip bound for scaled p0 channel
    cfg_step2["y_max_after_scale"] = y_max_after_scale  # clip bound for scaled p1 channel
    cfg_step2["iter_p7_ts"] = dt
    cfg_step2["hidden_dims"] = hidden_dims

    if cfg_step2["k_source_mode"] == "imported_nn":
        source_step1_pkl = out_dir / step1_run_name_pkl_arr[iter_idx - 1]
        if not source_step1_pkl.is_file():
            raise FileNotFoundError(f"Missing source Step1 pkl: {source_step1_pkl}")
        cfg_step2["source_step1_pkl"] = str(source_step1_pkl)
        print("m_fir is not allowed to change for this mode")
        cfg_step2["strict_source_match"] = True

    if cfg_step2["k_source_mode"] == "poly_lifting":
        cfg_step2["poly_order"] = int(3) # poly_order supports 0..3; 0 means pure FIR with constant lifting k(t)=1
        print('Can not select n_branch in poly_lifting mode since it is determined by other variables')
        cfg_step2["m_fir"] = m_fir

    cfg_step2["is_passive"] = is_passive

    cfg_step2 = step2_min_core.recheck_cfg_min(cfg_step2, verbose=True)
    print("[step2_min] start")
    step2_result = step2_min_core.run_step2_min(cfg_step2)
    print("[step2_min] done")
    print("solver_status:", step2_result["solver_status"])
    print("opt_value:", step2_result["opt_value"])
    print("train_mse:", step2_result["train_mse"])
    print("val_mse:", step2_result["val_mse"])
    print("test_mse:", step2_result["test_mse"])
    print("min_decay_margin:", float(np.min(step2_result["decay_margin"])))
    print("min_passivity_margin:", float(np.min(step2_result["passivity_min_margin"])))
    print("pkl:", step2_result["pkl_path"])
    print("mat:", step2_result["mat_path"])
    print(" ")
    print(" ")
    print(" ")


    cfg_step1 = train_core.build_default_config()
    cfg_step1["save_full_diagnostics"] = save_full_diagnostics_ON # whether save all details 

    cfg_step1["run_name"] = step1_run_name_arr[iter_idx] # 

    cfg_step1["fir_source_type"] = "step2_pkl"
    prev_step2_pkl = Path(step2_result["pkl_path"])
    if not prev_step2_pkl.is_file():
        raise FileNotFoundError(f"Missing prev_step2_pkl: {prev_step2_pkl}")
    cfg_step1["fir_source_step2_pkl"] =  str(prev_step2_pkl)

    g_bank_jm = np.asarray(step2_result["g_bank"], dtype=float)  # shape (J,M)
    cfg_step1["n_branch"] = int(g_bank_jm.shape[0])
    cfg_step1["m_fir"] = int(g_bank_jm.shape[1])

    cfg_step1["x_max"] = None  # None or float, the cap after feature normalisation. Should be 1.0 for poly. But None for NN. 
    cfg_step1["feature_norm_mode"] = feature_norm_mode # 'none' or 'zscore'
    cfg_step1["fixed_uy_scale"] = fixed_uy_scale
    cfg_step1["rebuild_p7_from_uy"] = rebuild_p7_from_uy 
    cfg_step1["u_scale_fixed"] = u_scale_fixed
    cfg_step1["y_scale_fixed"] = y_scale_fixed
    cfg_step1["uy_scale_method"] = uy_scale_method  # "softsign" => x/sqrt(x^2+s^2), "divide" => x/s
    cfg_step1["u_max_after_scale"] = u_max_after_scale # clip bound for scaled p0 channel
    cfg_step1["y_max_after_scale"] = y_max_after_scale  # clip bound for scaled p1 channel
    cfg_step1["hidden_dims"] = hidden_dims  # num of neurons in hidden layer 1 and 2
    cfg_step1["max_epochs"] = 1000 
    cfg_step1["early_stopping_patience"] = 300

    cfg_step1["active_dims"] = active_dims # if just one dimension, then use (1,) or (2,) or (0,)
    cfg_step1["delay_steps_by_dim"] = delay_steps_by_dim
    cfg_step1["train_val_test_split"] = (300, 100, 100) 
    cfg_step1["batch_size"] = 8
    cfg_step1["mlp_hidden_activation"] = "tanh"  # scalar string: activation for hidden layers.
    cfg_step1["mlp_output_activation"] = "tanh"  # scalar string: activation for output layer k_j(t).
    cfg_step1["dt"] = dt

    if last_bptt_on and iter_idx == max_iteration - 1:
        cfg_step1["skip_open_loop_training"] = True 
        cfg_step1["bptt_finetune_enable"] = True  
        cfg_step1["bptt_max_epochs"] = 100
        cfg_step1["bptt_learning_rate"] = 5e-5
        cfg_step1["bptt_batch_size"] = 8
        cfg_step1["bptt_early_stopping"] = True
        cfg_step1["bptt_early_stopping_patience"] = 50
        cfg_step1["bptt_grad_clip_norm"] = 1.0

        cfg_step1["bptt_lr_scheduler"] = "onecycle"
        cfg_step1["bptt_lr_decay_factor"] = 0.5
        cfg_step1["bptt_lr_decay_patience"] = 5
        cfg_step1["bptt_lr_gamma"] = 0.95
        cfg_step1["bptt_min_learning_rate"] = 1e-6
        cfg_step1["bptt_warmup_epochs"] = 5
        cfg_step1["bptt_onecycle_pct_start"] = 0.1
        cfg_step1["bptt_onecycle_div_factor"] = 10.0
        cfg_step1["bptt_onecycle_final_div_factor"] = 100.0
    else:
        cfg_step1["skip_open_loop_training"] = False 
        cfg_step1["bptt_finetune_enable"] = False  
        cfg_step1["bptt_max_epochs"] = 100
        cfg_step1["bptt_learning_rate"] = 5e-5
        cfg_step1["bptt_batch_size"] = 4
        cfg_step1["bptt_early_stopping"] = True
        cfg_step1["bptt_early_stopping_patience"] = 50
        cfg_step1["bptt_grad_clip_norm"] = 1.0
    

    # source_step1_pkl = out_dir / step1_run_name_pkl_arr[iter_idx - 1]
    # if not source_step1_pkl.is_file():
    #     raise FileNotFoundError(f"Missing source theta_N pkl: {source_step1_pkl}")
    if iter_idx > 0:
        if NN_para_inher_double_iter == False:
            initial_model_state_dict = None
        else:
            prev_step1_nn_run_name = step1_run_name_arr[iter_idx-1] # 
            baseline_pkl = out_dir / f"{prev_step1_nn_run_name}.pkl"

            with baseline_pkl.open("rb") as f:
                baseline_step1 = pickle.load(f)
                initial_model_state_dict = baseline_step1["model_state_dict"]
    else:
        initial_model_state_dict = None

    
    print("[min-Step1] Starting")
    step1_result = train_core.run_from_mat_file(
        mat_path=str(mat_path),
        out_dir=str(out_dir),
        run_name=cfg_step1["run_name"],
        cfg=cfg_step1,
        initial_model_state_dict=initial_model_state_dict)
    
    print("[min-Step1] Done")
    print("device:", step1_result["device"])
    print("best_epoch:", step1_result["best_epoch"])
    print("best_val_loss:", step1_result["best_val_loss"])
    print("pkl:", step1_result["pkl_path"])
    print("mat:", step1_result["mat_path"])
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
