# Experiment 2: Section VI.B

This directory contains the code, data, and selected saved results for the
Section VI.B robot-arm experiment. The experiment compares the proposed NFIR
method with FIR and MATLAB grey-box baselines.

Table II in the paper reports Fit metrics on the robot-arm trajectories. The
analysis script currently plots the closed-loop Fit distributions for `ue1`,
`uv1`, and the combined `uv2`/`uv3` test trajectories. The saved result files
required for this comparison are included in `Results/`.

## Directory Structure

- `Core_code/`: shared NFIR training, feature construction, optimization, and
  data I/O code.
- `Training_data/`: robot-arm MATLAB datasets used by the Python and MATLAB
  scripts.
- `Results/`: provided result files for NFIR, FIR, and grey-box baselines.
- `Other_models_in_Exp2/`: baseline scripts for FIR and MATLAB grey-box
  identification.
- `utility_functions/`: MATLAB helper functions used by the Table II analysis.
- `train_NFIR_Table_II.py`: Python driver for reproducing the selected NFIR
  Table II case.
- `script_NFIR_Table_II_results_analysis.m`: MATLAB script that loads the
  provided result files and computes Table II metrics.

## Reproduce The NFIR Table II Case

Run the selected NFIR training case from the repository root:

```bash
python Exp2/train_NFIR_Table_II.py
```

The driver calls:

```text
Exp2.Core_code.Exp2_NFIR
```

and writes outputs to:

```text
Exp2/Results/
```

The full NFIR run can take a long time. The saved outputs used by the paper
comparison are already included in `Exp2/Results/`.

## Baseline Models

The FIR baseline can be run from the repository root:

```bash
python Exp2/Other_models_in_Exp2/Exp2_FIR.py
```

The MATLAB grey-box baseline is stored in:

```text
Exp2/Other_models_in_Exp2/Exp2_Greybox.m
```

It uses MATLAB System Identification tooling and saves:

```text
Exp2/Results/Data_Exp2_Greybox.mat
```

## Table II Analysis

From the repository root, run:

```matlab
run('Exp2/script_NFIR_Table_II_results_analysis.m')
```

Or from inside the `Exp2` directory:

```matlab
run('script_NFIR_Table_II_results_analysis.m')
```

The analysis script loads the provided NFIR, FIR, and grey-box result files
from `Exp2/Results/` and adds `Exp2/utility_functions/` to the MATLAB path.
It produces closed-loop-only box plots titled:

```text
ue1 fitting
uv1 fitting
uv2 and uv3 fitting
```

Validation-based early stopping is not used in this Exp2 NFIR run. Therefore
`uv1` is not used to choose the model during training and remains unseen by
the training process.

## Notes On Dependencies

This repository does not currently provide a pinned Python environment file.
The Python code uses packages including NumPy, SciPy, PyTorch, CVXPY, and
MOSEK. MATLAB is used for the Table II analysis and the grey-box baseline.
