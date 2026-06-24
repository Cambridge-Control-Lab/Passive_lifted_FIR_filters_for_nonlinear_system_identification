from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io


GREYBOX_RESULT_NAME = "Matlab_grey"  # scalar str: variable name in Data_Exp2_Greybox.mat.
FIR_RESULT_NAME = "run_Exp2_FIR_final_m500_zc50_train"  # scalar str: FIR result variable/file stem.
NFIR_RESULT_NAME = (
    "run_Exp2_150_final_h8_ad01_m500_b3_nonoise_pas_ite4_nobptt_"
    "l220_l2NN5_firParallOn_ep20_zc50_neoit_s13_train"
)  # scalar str: NFIR result variable/file stem.
# The robot-arm batches have different non-zero initial conditions. The early
# part of each trajectory is strongly affected by those initial conditions, so
# Exp2/train_NFIR_Table_II.py uses zero_cost_first_n = 50 during training. The
# MATLAB Table II script uses n_sample_out = 50, which means MATLAB rows 50:end.
# In zero-based Python slicing, the same row start is index 49.
N_SAMPLE_OUT = 49  # scalar int: Python row start matching MATLAB 50:end for Table II metrics.
TABLE_II_COL_LABELS = ("ue", "uv1", "uv2", "uv3")  # shape (4,), Table II column labels.


def find_exp2_dir() -> Path:
    """Locate the Exp2 folder when run from the repo root or from Exp2."""
    cwd = Path.cwd()  # scalar Path: current working directory.
    script_dir = Path(__file__).resolve().parent  # scalar Path: folder containing this script.
    candidate_dirs = (
        cwd,
        cwd / "Exp2",
        script_dir,
    )  # shape (3,): possible Exp2 folders.

    for candidate_dir in candidate_dirs:
        results_dir = candidate_dir / "Results"  # scalar Path: candidate Results folder.
        training_dir = candidate_dir / "Training_data"  # scalar Path: candidate Training_data folder.
        if results_dir.is_dir() and training_dir.is_dir():
            return candidate_dir

    raise FileNotFoundError("Run this script from the repository root or from the Exp2 folder.")


def load_mat_struct(mat_path: Path, variable_name: str):
    """Load one scalar MATLAB struct from a .mat file."""
    mat_data = scipy.io.loadmat(
        str(mat_path),
        squeeze_me=False,
        struct_as_record=False,
    )  # dict[str,object]: MATLAB file contents.
    struct_arr = mat_data[variable_name]  # np.ndarray, shape (1,1), MATLAB struct array.
    return struct_arr[0, 0]


def new_metrics_by_traj(y_true_mat: np.ndarray, y_pred_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute Fit[%] and MSE for each trajectory column."""
    y_true_tb = np.asarray(y_true_mat, dtype=float)  # np.ndarray, shape (T,B), true output.
    y_pred_tb = np.asarray(y_pred_mat, dtype=float)  # np.ndarray, shape (T,B), predicted output.
    err_tb = y_true_tb - y_pred_tb  # np.ndarray, shape (T,B), prediction error.

    true_norm_b = np.linalg.norm(y_true_tb, ord=2, axis=0)  # np.ndarray, shape (B,), true-output norms.
    err_norm_b = np.linalg.norm(err_tb, ord=2, axis=0)  # np.ndarray, shape (B,), error norms.
    fit_b = 100.0 * (1.0 - err_norm_b / true_norm_b)  # np.ndarray, shape (B,), Fit[%] per trajectory.
    mse_b = np.mean(err_tb**2, axis=0)  # np.ndarray, shape (B,), MSE per trajectory.
    return fit_b, mse_b


def closed_loop_fit(data_struct, split_name: str) -> np.ndarray:
    """Compute closed-loop Fit[%] for one train/val/test split."""
    y_true_tb = getattr(data_struct, f"y_{split_name}_batch")  # np.ndarray, shape (T,B), true output.
    y_pred_tb = getattr(data_struct, f"y_pre_{split_name}_batch_cl")  # np.ndarray, shape (T,B), CL prediction.
    fit_b, _mse_b = new_metrics_by_traj(
        y_true_tb[N_SAMPLE_OUT:, :],
        y_pred_tb[N_SAMPLE_OUT:, :],
    )  # tuple[np.ndarray,np.ndarray], each shape (B,).
    return fit_b


def print_table_ii(row_labels: list[str], col_labels: tuple[str, ...], fit_mc: np.ndarray) -> None:
    """Print Table II values in the same order as the paper table."""
    n_model = len(row_labels)  # scalar int: number of model rows.
    n_col = len(col_labels)  # scalar int: number of Table II metric columns.
    row_label_width = max(len(label) for label in row_labels)  # scalar int: first-column width.
    value_width = 10  # scalar int: formatted value-column width.

    print("")
    print("TABLE II")
    print("INDUSTRIAL ROBOT MODEL-FIT PERFORMANCE.")
    print(" " * (row_label_width + 2) + "".join(f"{label:>{value_width}}" for label in col_labels))
    for i_model in range(n_model):
        row_text = f"{row_labels[i_model]:<{row_label_width}}  "  # scalar str: model label cell.
        for i_col in range(n_col):
            value_text = f"{fit_mc[i_model, i_col]:.2f}%"  # scalar str: Fit (%) table cell.
            row_text += f"{value_text:>{value_width}}"
        print(row_text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive plot windows.")
    args = parser.parse_args()

    if bool(args.no_show):
        import matplotlib

        matplotlib.use("Agg")

    exp2_dir = find_exp2_dir()  # scalar Path: Exp2 folder.
    results_dir = exp2_dir / "Results"  # scalar Path: Exp2 Results folder.

    d_fir = load_mat_struct(results_dir / f"{FIR_RESULT_NAME}.mat", FIR_RESULT_NAME)  # scalar mat_struct: FIR result.
    d_grey = load_mat_struct(results_dir / "Data_Exp2_Greybox.mat", GREYBOX_RESULT_NAME)  # scalar mat_struct: grey-box result.
    d_nfir = load_mat_struct(results_dir / f"{NFIR_RESULT_NAME}.mat", NFIR_RESULT_NAME)  # scalar mat_struct: NFIR result.

    labels = ["FIR", "Grey-box", "NFIR"]  # shape (3,), labels for plotted models.
    data_list = [d_fir, d_grey, d_nfir]  # shape (3,), loaded result structs.

    fit_train_cl = [closed_loop_fit(data_struct, "train") for data_struct in data_list]  # list, each shape (B_i,).
    fit_val_cl = [closed_loop_fit(data_struct, "val") for data_struct in data_list]  # list, each shape (B_i,).
    fit_test_cl = [closed_loop_fit(data_struct, "test") for data_struct in data_list]  # list, each shape (B_i,).

    table_ii_fit_mc = np.zeros((len(labels), 4), dtype=float)  # np.ndarray, shape (n_model,4), Fit (%) table.
    for i_model in range(len(labels)):
        fit_train_b = fit_train_cl[i_model]  # np.ndarray, shape (1,), ue Fit (%).
        fit_val_b = fit_val_cl[i_model]  # np.ndarray, shape (1,), uv1 Fit (%).
        fit_test_b = fit_test_cl[i_model]  # np.ndarray, shape (2,), uv2 and uv3 Fit (%).
        table_ii_fit_mc[i_model, :] = [
            fit_train_b[0],
            fit_val_b[0],
            fit_test_b[0],
            fit_test_b[1],
        ]  # np.ndarray row, shape (4,).

    print(f"Exp2 dir: {exp2_dir}")
    print_table_ii(labels, TABLE_II_COL_LABELS, table_ii_fit_mc)


    if not bool(args.no_show):
        import matplotlib.pyplot as plt

        plt.show()
    else:
        import matplotlib.pyplot as plt

        plt.close("all")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
