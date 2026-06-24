# Experiment 2: Section VI.B

This directory contains the code, data, and selected saved results for the
Section VI.B of the paper. The experiment compares the proposed NFIR
method with FIR and MATLAB grey-box baselines.

Table II in the paper reports Fit metrics on the robot-arm trajectories. The
Python analysis script shows the Fit distributions for `ue1`,
`uv1`,  `uv2` and `uv3` test trajectories. The saved result files
required for this comparison are included in `Results/`.

## Directory Structure

- `Core_code/`: shared NFIR training, feature construction, optimization, and
  data I/O code.
- `Training_data/`: robot-arm datasets used by the Python training scripts.
- `Results/`: provided result files for NFIR, FIR, and grey-box baselines.
- `Other_models_in_Exp2/`: optional baseline model code.
- `train_NFIR_Table_II.py`: Python script for reproducing the selected NFIR
  Table II case.
- `script_NFIR_Table_II_results_analysis.py`: Python script that loads the
  provided result files and computes Table II metrics.

## Training models 

Before running this experiment, create and activate the conda environment
following the setup instructions in the repository-level [README.md](../README.md).

Run the selected NFIR training case from the repository root:

```bash
python Exp2/train_NFIR_Table_II.py
```

`Exp2/train_NFIR_Table_II.py` runs the module:

```text
Exp2.Core_code.Exp2_NFIR
```

`Exp2.Core_code.Exp2_NFIR` writes outputs to:

```text
Exp2/Results/
```

The saved outputs used by the paper
comparison are already included in `Exp2/Results/`.

For the selected Table II case, `Exp2/train_NFIR_Table_II.py` runs four outer
iterations.
Each iteration saves one `theta_G` FIR-bank optimization result and one
`theta_N` neural-lifting result to `Exp2/Results/`. 


The Table II model-fit analysis uses the final `theta_N` result:

```text
Exp2/Results/run_Exp2_150_final_h8_ad01_m500_b3_nonoise_pas_ite4_nobptt_l220_l2NN5_firParallOn_ep20_zc50_neoit_s13_train.mat
```

By default, the training code uses `mps`. MPS is PyTorch's Apple Metal
backend for Apple Silicon GPUs, such as M1, M2, M3, and newer Apple Silicon
Macs. The provided Exp2 NFIR result files in this folder were generated with
`mps`.

The supported device settings are:

- `mps`: Apple Silicon GPU through PyTorch MPS. This is the default and the
  tested setting for the provided NFIR results.
- `cpu`: CPU-only execution. This should work on any machine, but is slower.
- `cuda`: NVIDIA GPU through CUDA-enabled PyTorch. This should work on a
  suitable CUDA machine, but has not been tested for this release.

To change the device, edit:

```text
Exp2/Core_code/theta_N_parallel_fir_core.py
```

Inside `build_default_config()`, find the device settings:

```python
# cfg["device"] = "cpu"
cfg["device"] = "mps"
```

For CPU, change them to:

```python
cfg["device"] = "cpu"
# cfg["device"] = "mps"
```

Then run the same command:

```bash
python Exp2/train_NFIR_Table_II.py
```

For CUDA, set the same device field to:

```python
cfg["device"] = "cuda"
```
But cuda is not tested in this release. 

### Baseline Models

The FIR baseline can be run from the repository root:

```bash
python Exp2/Other_models_in_Exp2/Exp2_FIR.py
```

It saves the FIR baseline results to:

```text
Exp2/Results/run_Exp2_FIR_final_m500_zc50.pkl
Exp2/Results/run_Exp2_FIR_final_m500_zc50_train.mat
```

The grey-box baseline is the MATLAB script:

```text
Exp2/Other_models_in_Exp2/Exp2_Greybox.m
```

Run it from MATLAB with the repository root on the MATLAB path, or open the
script in MATLAB and run it directly. It saves the grey-box comparison result
to:

```text
Exp2/Results/Data_Exp2_Greybox.mat
```

## Getting Table II

From the repository root, run:

```bash
python Exp2/script_NFIR_Table_II_results_analysis.py
```

Or from inside the `Exp2` directory:

```bash
python script_NFIR_Table_II_results_analysis.py
```

The analysis script loads these provided result files from `Exp2/Results/`:

```text
Data_Exp2_Greybox.mat
run_Exp2_FIR_final_m500_zc50_train.mat
run_Exp2_150_final_h8_ad01_m500_b3_nonoise_pas_ite4_nobptt_l220_l2NN5_firParallOn_ep20_zc50_neoit_s13_train.mat
```

It prints the Table II model-fit values with columns `ue`, `uv1`, `uv2`, and
`uv3`. 

Validation-based early stopping is not used in this Exp2 NFIR run. Therefore
`uv1` is not used to choose the model during training and remains unseen by
the training process.
