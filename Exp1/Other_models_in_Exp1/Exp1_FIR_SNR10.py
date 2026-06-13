from __future__ import annotations
from pathlib import Path
import sys
import pickle

SCRIPT_DIR = Path(__file__).resolve().parent  # scalar Path: baseline script folder.
EXP1_DIR = SCRIPT_DIR.parent  # scalar Path: Exp1 folder.
REPO_ROOT = EXP1_DIR.parent  # scalar Path: repository root.
DATA_DIR = EXP1_DIR / "Training_data"  # scalar Path: training data folder.
RESULTS_DIR = EXP1_DIR / "Results"  # scalar Path: output results folder.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
print("EXP1_DIR is ", EXP1_DIR)
mat_path = DATA_DIR / "Data_M_NLdamper_500B_OneCart_SNR10.mat"  # Set input MAT data path. Training data. 
out_dir = RESULTS_DIR  # Set output folder path where output is exported. 
if not mat_path.is_file():
    raise FileNotFoundError(f"Missing MAT data file: {mat_path}")
out_dir.mkdir(parents=True, exist_ok=True)

from Exp1.Core_code import theta_G_core as step2_min_core
import numpy as np


"""Configuration note:
    When Step1 is run without initial_model_state_dict, the NN initialization
    is deterministic for the same random seed.
"""

# Hand-edit only this integer for pilot runs.
dt = 0.02 
hidden_dims = (256, 256)
active_dims = (0,1,2,3,6)
delay_steps_by_dim = {}
fixed_uy_scale = True
rebuild_p7_from_uy = True
u_scale_fixed = float(16.5)
y_scale_fixed = float(2.94)
uy_scale_method = "divide"
u_max_after_scale =  100000.0 
y_max_after_scale =  100000.0 
feature_norm_mode = "zscore"
m_fir = 50
save_full_diagnostics_ON = False
NN_para_inher_double_iter = True # If True, NN initial parameters come from the previous iteration.
"""Configuration note:
    When Step1 is run without initial_model_state_dict, the NN initialization
    is deterministic for the same random seed.
"""

mode = "poly_lifting"
cfg_step2 = step2_min_core.build_default_config_min(mode)
cfg_step2["save_full_diagnostics"] = save_full_diagnostics_ON # whether save all details 
cfg_step2["k_source_mode"] = mode
cfg_step2["run_name"] = "run_Exp1_FIR_SNR10"# the file name of final output data after optimisation
# cfg_step2["run_name"] = "run_M9_FIRonly_train70batches"# the file name of final output data after optimisation

cfg_step2["out_dir"] = str(out_dir)  # which folder in this workspace to store the output results
cfg_step2["source_data_mat"] = str(mat_path) #matlab file for training data

cfg_step2["active_dims"] = active_dims # if just one dimension, then use (1,) or (2,) or (0,)
cfg_step2["delay_steps_by_dim"] = delay_steps_by_dim
cfg_step2["n_refine_iter"] = 0
cfg_step2["iter_yhat_add_noise"] = False 
cfg_step2["train_val_test_split"] = (300, 100, 100) 
# cfg_step2["train_val_test_split"] = (70, 330, 100) 
cfg_step2["imported_split_source"] = "cfg"   # "step1": use Step1 saved split; "cfg": use Step2 train_val_test_split.
cfg_step2["ms_passivity"] = 10000 
cfg_step2["x_max"] = 1.0

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

if cfg_step2["k_source_mode"] == "poly_lifting":
    cfg_step2["poly_order"] = int(0) # poly_order supports 0..3; 0 means pure FIR with constant lifting k(t)=1
    print('Can not select n_branch in poly_lifting mode since it is determined by other variables')
    cfg_step2["m_fir"] = m_fir

cfg_step2["is_passive"] = True

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
