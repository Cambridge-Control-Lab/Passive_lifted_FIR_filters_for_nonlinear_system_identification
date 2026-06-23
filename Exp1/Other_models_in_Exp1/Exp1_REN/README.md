# Exp1 REN Baseline Reproducibility Guide

These scripts reproduce the REN baseline for Section VI.A / Fig. 4 using the
Julia toolbox [`RobustNeuralNetworks.jl`](https://github.com/acfr/RobustNeuralNetworks.jl/tree/main).

The toolbox code is not stored in this repository. The commands below clone the
toolbox into this folder, install its Julia example environment, copy the Exp1
REN scripts into the toolbox, then train and evaluate the clean-data and noisy
SNR10 REN baselines.

Generated REN model and evaluation files are saved to:

```text
Exp1/Results/
```

The repository already contains the final REN `.mat` result files. Run the full
Julia workflow only if you want to regenerate the REN `.bson` model files and
evaluation `.mat` files.

## Tested Environment

The workflow below was tested with:

```text
macOS 26.5.1
Julia 1.12.6
RobustNeuralNetworks.jl commit 1d0b5ab8c326719c67e2adb85d4db01e742883d9
```

Other Julia versions may work, but the versions above are the known tested
setup.

## Full Reproduction Commands

Replace `/path/to/Passive_lifted_FIR_filters_for_nonlinear_system_identification`
with the absolute path to this repository on your computer.


Then run:

```bash
set -e

cd /path/to/Passive_lifted_FIR_filters_for_nonlinear_system_identification/Exp1/Other_models_in_Exp1/Exp1_REN

if test ! -d RobustNeuralNetworks.jl; then
  git clone https://github.com/acfr/RobustNeuralNetworks.jl.git
else
  echo "Using existing RobustNeuralNetworks.jl folder."
fi

cd RobustNeuralNetworks.jl
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=examples -e 'using Pkg; Pkg.add("MAT"); Pkg.instantiate()'
cd ..

cp REN_load_data.jl \
   REN_train_exp1.jl \
   REN_train_exp1_noise.jl \
   REN_eval_exp1.jl \
   RobustNeuralNetworks.jl/examples/src/

export NFIR_EXP1_DIR="$(cd ../.. && pwd)"

cd RobustNeuralNetworks.jl

julia --project=examples examples/src/REN_train_exp1.jl
julia --project=examples examples/src/REN_eval_exp1.jl

julia --project=examples examples/src/REN_train_exp1_noise.jl
julia --project=examples examples/src/REN_eval_exp1.jl --snr10
```

Command notes:

- `set -e` stops the script as soon as any command fails.
- The `git clone` block downloads `RobustNeuralNetworks.jl` only if the folder
  does not already exist.
- `Pkg.develop(path=".")` makes the Julia example environment use the local
  toolbox checkout.
- `Pkg.add("MAT")` is needed because `REN_load_data.jl` uses `using MAT`.
- `NFIR_EXP1_DIR` points the Julia scripts back to this repository's `Exp1`
  folder, so outputs go to `Exp1/Results/`.

The first Julia installation command can take several minutes because Julia
downloads and precompiles the toolbox dependencies.

## Script Roles

- [REN_train_exp1.jl](REN_train_exp1.jl): trains the clean-data REN model used
  for the left panel of Fig. 4.
- [REN_train_exp1_noise.jl](REN_train_exp1_noise.jl): trains the noisy-data
  SNR10 REN model used for the right panel of Fig. 4.
- [REN_load_data.jl](REN_load_data.jl): loads and prepares the Exp1 training
  data for the training scripts.
- [REN_eval_exp1.jl](REN_eval_exp1.jl): loads a trained `.bson` model and writes
  inference/evaluation outputs to `.mat`.

## Expected Outputs

Clean-data training:

```bash
julia --project=examples examples/src/REN_train_exp1.jl
```

saves:

```text
Exp1/Results/run_Exp1_REN_model.bson
```

Clean-data evaluation:

```bash
julia --project=examples examples/src/REN_eval_exp1.jl
```

saves:

```text
Exp1/Results/run_Exp1_REN_train.mat
Exp1/Results/run_Exp1_REN_training_cost_vs_epoch.svg
```

Noisy SNR10 training:

```bash
julia --project=examples examples/src/REN_train_exp1_noise.jl
```

saves:

```text
Exp1/Results/run_Exp1_REN_SNR10_model.bson
```

Noisy SNR10 evaluation:

```bash
julia --project=examples examples/src/REN_eval_exp1.jl --snr10
```

saves:

```text
Exp1/Results/run_Exp1_REN_SNR10_train.mat
Exp1/Results/run_Exp1_REN_SNR10_training_cost_vs_epoch.svg
```
