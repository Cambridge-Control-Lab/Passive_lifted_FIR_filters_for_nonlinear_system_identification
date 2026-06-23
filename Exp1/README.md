# Experiment 1: Section VI.A

This folder contains the code, data, and selected saved results for the
Section VI.A of the paper.

Fig. 4 in the paper reports the Fit metric (Equation 21 of the paper) on unseen test trajectories for
different model classes. The files provided here are intended to reproduce
and inspect the FB-BP noisy case in the right panel of Fig. 4, and to provide
the baseline model code and saved result files used for comparison.

In the noisy training condition, Gaussian white noise with SNR = 10 dB is
added to the output during training. The input signals are noise-free in both
the training and test datasets. When the Fit metric is computed for the noisy
case, the model prediction is compared with the clean ground-truth output.

## Folder Structure

- `Core_code/`: shared NFIR training, feature construction, optimization, and
  data I/O code.
- `Training_data/`: clean and SNR10 training datasets for the nonlinear
  mass-spring-damper system.
- `Results/`: provided MATLAB result files. These include FB-BP noisy and
  noise-free results, plus baseline result files for FIR, MLP, REN, and
  available N4SID outputs.
- `Other_models_in_Exp1/`: baseline scripts for FIR, MLP, N4SID, and REN.
  The REN baseline has its own README because it depends on a separate Julia
  toolbox.
- `train_FB_BP_noisy_Fig4.py`: Python driver for reproducing the FB-BP noisy
  case.
- `script_FB_BP_noisy_results_analysis.py`: simple Python script that plots the
  FB-BP box in the right panel of Fig. 4 from the provided result files.

## Reproduce The FB-BP Noisy Case

Run the FB-BP noisy training case from the repository root:

```bash
python Exp1/train_FB_BP_noisy_Fig4.py
```

The script train_FB_BP_noisy_Fig4.py calls:

```text
Exp1.Core_code.Exp1_NFIR_training_code
```

and writes outputs to:

```text
Exp1/Results/
```

## Plot The Provided FB-BP Noisy Result

The provided Python script plots only the FB-BP noisy box used in the right
panel of Fig. 4.

From the repository root:

```bash
python Exp1/script_FB_BP_noisy_results_analysis.py
```

Or from inside the `Exp1` folder:

```bash
python script_FB_BP_noisy_results_analysis.py
```

The script loads both the noise-free and noisy FB-BP result files. The clean
output trajectory from the noise-free result is used as the ground-truth
reference for the Fit metric, while the prediction comes from the model
trained with noisy output data.

## Configuration Variants

The default configuration in `train_FB_BP_noisy_Fig4.py` corresponds to the
FB-BP noisy case:

```python
"active_dims": "0,1"
"is_passive": "1"
```

To run the FF/feedforward case, change:

```python
"active_dims": "0"
```

To run a non-passive `-np` case, change:

```python
"is_passive": "0"
```


## Baseline Models

Code for the FIR, MLP, N4SID, and REN baselines is stored in:

```text
Exp1/Other_models_in_Exp1/
```

The saved baseline `.mat` files are already provided in:

```text
Exp1/Results/
```

The REN scripts require the external `RobustNeuralNetworks.jl` toolbox. See
`Other_models_in_Exp1/Exp1_REN/README.md` for the REN-specific workflow.
