# Exp1 REN Baseline

These scripts reproduce the REN baseline used in Exp1. They require the separate Julia toolbox `RobustNeuralNetworks.jl`, so they are not expected to run from this repository by themselves.

For simplicity, the final REN evaluation result files have already been copied into this repository's `Exp1/Results/` folder. You only need to run the Julia workflow below if you want to regenerate the REN model and evaluation outputs.

## 1. Install The Julia Toolbox

Download or clone `RobustNeuralNetworks.jl` to a separate directory on your computer.

```bash
git clone https://github.com/acfr/RobustNeuralNetworks.jl.git
cd RobustNeuralNetworks.jl
```

From the toolbox root, install the example environment:

```bash
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

The first run can take several minutes because Julia downloads and precompiles the example dependencies.

## 2. Copy The Exp1 REN Scripts

Copy these four files from this repository:

```text
Exp1/Other_models_in_Exp1/Exp1_REN/REN_load_data.jl
Exp1/Other_models_in_Exp1/Exp1_REN/REN_train_exp1.jl
Exp1/Other_models_in_Exp1/Exp1_REN/REN_train_exp1_noise.jl
Exp1/Other_models_in_Exp1/Exp1_REN/REN_eval_exp1.jl
```

Paste them into the toolbox folder:

```text
RobustNeuralNetworks.jl/examples/src/
```

## 3. Point The Scripts To This Repository

Set `NFIR_EXP1_DIR` to the `Exp1` folder in this open-source repository:

```bash
export NFIR_EXP1_DIR="/path/to/c_final_opensource/Exp1"
```

The scripts use:

```text
$NFIR_EXP1_DIR/Training_data
$NFIR_EXP1_DIR/Results
```

## 4. Run The REN Baseline

Run all commands from the `RobustNeuralNetworks.jl` toolbox root.

Clean-data REN:

```bash
julia --project=examples examples/src/REN_train_exp1.jl
julia --project=examples examples/src/REN_eval_exp1.jl
```

SNR10 REN:

```bash
julia --project=examples examples/src/REN_train_exp1_noise.jl
julia --project=examples examples/src/REN_eval_exp1.jl --snr10
```

The training scripts save:

```text
Exp1/Results/run_Exp1_REN_model.bson
Exp1/Results/run_Exp1_REN_SNR10_model.bson
```

The evaluation script saves:

```text
Exp1/Results/run_Exp1_REN_train.mat
Exp1/Results/run_Exp1_REN_SNR10_train.mat
```
