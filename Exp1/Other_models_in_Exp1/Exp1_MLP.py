from __future__ import annotations

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent  # scalar Path: baseline script folder.
EXP1_DIR = SCRIPT_DIR.parent  # scalar Path: Exp1 folder.
REPO_ROOT = EXP1_DIR.parent  # scalar Path: repository root.
DATA_DIR = EXP1_DIR / "Training_data"  # scalar Path: training data folder.
RESULTS_DIR = EXP1_DIR / "Results"  # scalar Path: output results folder.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Exp1.Core_code import direct_u_mlp_torch_core as mlp_core


def build_Exp1_cfg() -> dict:
    """Build Exp1 direct-u PyTorch MLP config."""
    cfg = mlp_core.build_default_config()
    cfg["run_name"] = "run_Exp1_MLP"
    cfg["u_delay_steps"] = 49
    cfg["hidden_layer_sizes"] = (10, 10)
    cfg["learning_rate"] = 1e-3
    cfg["weight_decay"] = 1e-5
    cfg["max_epochs"] = 1000
    cfg["batch_size"] = 2000 # 2048/250 = 8 trajectories per mini-batch.
    cfg["early_stopping"] = True
    cfg["early_stopping_patience"] = 300
    cfg["random_seed"] = 1
    cfg["train_val_test_split"] = (300, 100, 100)
    cfg["u_scale_fixed"] = 16.5
    cfg["uy_scale_method"] = "divide"
    cfg["feature_norm_mode"] = "zscore"
    cfg["target_norm_mode"] = "zscore"
    cfg["x_max"] = None
    cfg["device"] = "mps"
    cfg["verbose"] = True
    cfg["log_every"] = 50
    return cfg


def main() -> None:
    mat_path = DATA_DIR / "Data_M_NLdamper_500B_OneCart.mat"
    out_dir = RESULTS_DIR
    cfg = build_Exp1_cfg()

    print("[direct-torch] Starting")
    print("EXP1_DIR:", EXP1_DIR)
    print("mat_path:", mat_path)
    print("run_name:", cfg["run_name"])
    result = mlp_core.run_from_mat_file(
        mat_path=mat_path,
        out_dir=out_dir,
        run_name=cfg["run_name"],
        cfg=cfg,
    )
    print("[direct-torch] Done")
    print("best_epoch:", result["best_epoch"])
    print("best_val_loss:", result["best_val_loss"])
    print("train_mse:", result["train_mse"])
    print("val_mse:", result["val_mse"])
    print("test_mse:", result["test_mse"])
    print("mlp_train_time_sec:", result["mlp_train_time_sec"])
    print("nn_train_time_sec:", result["nn_train_time_sec"])
    print("pkl:", result["pkl_path"])
    print("mat:", result["mat_path"])


if __name__ == "__main__":
    main()
