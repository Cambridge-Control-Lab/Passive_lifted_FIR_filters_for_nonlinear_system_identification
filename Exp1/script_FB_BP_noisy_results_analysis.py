from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io


NONOISE_RESULT_NAME = (
    "run_NFIR_0_h4_ad01_m50_b10_nonoise_pas_ite5_lastbptt_l21_neoit_s14_train"
)  # scalar str: variable name and file stem for the noise-free result.
NOISE_RESULT_NAME = (
    "run_NFIR_0_h4_ad01_m50_b10_noise_pas_ite5_lastbptt_l21_neoit_s14_train"
)  # scalar str: variable name and file stem for the noisy result.


def find_exp1_dir() -> Path:
    """Locate the Exp1 folder when run from the repo root or from Exp1."""
    cwd = Path.cwd()  # scalar Path: current working directory.
    script_dir = Path(__file__).resolve().parent  # scalar Path: folder containing this script.
    candidate_dirs = (
        cwd,
        cwd / "Exp1",
        script_dir,
    )  # shape (3,): possible Exp1 folders.

    for candidate_dir in candidate_dirs:
        results_dir = candidate_dir / "Results"  # scalar Path: candidate Results folder.
        if results_dir.is_dir():
            return candidate_dir

    raise FileNotFoundError("Run this script from the repository root or from the Exp1 folder.")


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
    y_true_tb = np.asarray(y_true_mat, dtype=float)  # np.ndarray, shape (T,B), clean true output.
    y_pred_tb = np.asarray(y_pred_mat, dtype=float)  # np.ndarray, shape (T,B), predicted output.
    err_tb = y_true_tb - y_pred_tb  # np.ndarray, shape (T,B), prediction error.

    true_norm_b = np.linalg.norm(y_true_tb, ord=2, axis=0)  # np.ndarray, shape (B,), true-output norms.
    err_norm_b = np.linalg.norm(err_tb, ord=2, axis=0)  # np.ndarray, shape (B,), error norms.
    fit_b = 100.0 * (1.0 - err_norm_b / true_norm_b)  # np.ndarray, shape (B,), Fit[%] per trajectory.
    mse_b = np.mean(err_tb**2, axis=0)  # np.ndarray, shape (B,), MSE per trajectory.
    return fit_b, mse_b


def plot_fit_box(fit_test_cl_b: np.ndarray, show_plot: bool) -> None:
    """Plot the closed-loop noisy FB-BP Fit[%] box."""
    if not show_plot:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()  # Figure, Axes: one FB-BP boxplot figure.
    ax.boxplot(fit_test_cl_b.reshape(-1, 1))  # np.ndarray, shape (B,1), boxplot input.
    ax.set_ylabel("Fit[%]")
    ax.set_title("FB-BP noisy result")
    ax.grid(True)
    fig.tight_layout()

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive plot window.")
    args = parser.parse_args()

    exp1_dir = find_exp1_dir()  # scalar Path: Exp1 folder.
    results_dir = exp1_dir / "Results"  # scalar Path: Exp1 Results folder.
    result_file_nonoise = results_dir / f"{NONOISE_RESULT_NAME}.mat"  # scalar Path: noise-free result MAT file.
    result_file_noise = results_dir / f"{NOISE_RESULT_NAME}.mat"  # scalar Path: noisy result MAT file.

    d_nonoise = load_mat_struct(result_file_nonoise, NONOISE_RESULT_NAME)  # scalar mat_struct: noise-free result.
    d_noise = load_mat_struct(result_file_noise, NOISE_RESULT_NAME)  # scalar mat_struct: noisy result.

    # Paper metric: Fit[%] = 100 * (1 - ||y - y_hat||_2 / ||y||_2).
    # y is the clean ground-truth output. y_hat is the closed-loop prediction
    # from the model trained with noisy output data.
    y_test_tb = d_nonoise.y_test_batch  # np.ndarray, shape (T,B), clean ground-truth test output.
    y_pre_test_cl_tb = d_noise.y_pre_test_batch_cl  # np.ndarray, shape (T,B), noisy-model closed-loop output.
    fit_test_cl_b, mse_test_cl_b = new_metrics_by_traj(
        y_test_tb,
        y_pre_test_cl_tb,
    )  # tuple[np.ndarray,np.ndarray], each shape (B,).

    print(f"Exp1 dir: {exp1_dir}")
    print(f"fit_test_cl shape: {fit_test_cl_b.shape}")
    print(
        "fit_test_cl summary: "
        f"mean={np.mean(fit_test_cl_b):.6g}, "
        f"median={np.median(fit_test_cl_b):.6g}, "
        f"min={np.min(fit_test_cl_b):.6g}, "
        f"max={np.max(fit_test_cl_b):.6g}"
    )
    print(
        "mse_test_cl summary: "
        f"mean={np.mean(mse_test_cl_b):.6g}, "
        f"median={np.median(mse_test_cl_b):.6g}"
    )

    plot_fit_box(fit_test_cl_b, show_plot=not bool(args.no_show))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
