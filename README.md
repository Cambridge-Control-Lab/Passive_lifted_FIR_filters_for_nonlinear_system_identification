# Passive Lifted FIR Filters for Nonlinear System Identification

This repository contains code, data, and selected saved results for the paper
[Passive Lifted FIR Filters for Nonlinear System Identification](https://arxiv.org/abs/2508.05279)
by Zixing Wang and Fulvio Forni.

The code is intended for reproducing the main numerical experiments in the
paper and for inspecting the saved results used in the reported figures and
tables.

## Repository structure

- `Exp1/`: simulated nonlinear mass-spring-damper experiment from Section VI.A of the paper.
  See [Exp1/README.md](Exp1/README.md) for reproduction, device settings, and Fig. 4 result plotting.
- `Exp2/`: robot-arm experiment from Section VI.B of the paper. See [Exp2/README.md](Exp2/README.md) for
  the Table II workflow, baseline models, and analysis script.

## Getting started

Install conda first. Miniconda, Miniforge, and Anaconda all work; follow the
official conda installation instructions for your operating system:

```text
https://docs.conda.io/projects/conda/en/latest/user-guide/install/
```

Then clone the repository and run commands from the repository root:

```bash
git clone https://github.com/Cambridge-Control-Lab/Passive_lifted_FIR_filters_for_nonlinear_system_identification.git
cd Passive_lifted_FIR_filters_for_nonlinear_system_identification
```

Create the Python environment used for the experiments:

```bash
conda create -n nfir-env \
  -c conda-forge -c pytorch -c mosek \
  python=3.12 numpy=2.3 scipy=1.16 pytorch=2.5 cvxpy=1.5 mosek=11.0
```

Activate it before running the Python scripts:

```bash
conda activate nfir-env
```

MOSEK also requires a valid local license. Follow the MOSEK license setup
instructions for your installation before running the optimization scripts.

## Reproducing experiments

First create and activate the conda environment from the Getting started
section. Then run the experiment drivers from the repository root:

```bash
python Exp1/train_FB_BP_noisy_Fig4.py
python Exp2/train_NFIR_Table_II.py
```

Saved result files used for the paper comparisons are included in the
experiment `Results/` folders where available. Exp1 provides a Python plotting
script for the FB-BP noisy result; Exp2 provides a MATLAB analysis script for
Table II.

For detailed instructions, see:

- [Exp1/README.md](Exp1/README.md)
- [Exp2/README.md](Exp2/README.md)

## Dependencies

The Python experiments were tested with the following package versions:

- Python 3.12.11
- NumPy 2.3.1
- SciPy 1.16.0
- PyTorch 2.5.1
- CVXPY 1.5.3
- MOSEK 11.0.25

[Exp1/README.md](Exp1/README.md) describes the supported PyTorch device
settings: `mps`, `cpu`, and `cuda`. The provided Exp1 FB-BP results were
generated with `mps`.

MATLAB2025B or newer is required for the Exp2 Table II analysis script and
for the MATLAB baseline models.

## Citation

If you use this repository in academic work, please cite the paper:

```bibtex
@misc{wang2025passiveliftedFIR,
  title = {Passive Lifted FIR Filters for Nonlinear System Identification},
  author = {Zixing Wang and Fulvio Forni},
  year = {2025},
  eprint = {2508.05279},
  archivePrefix = {arXiv},
  primaryClass = {eess.SY},
  doi = {10.48550/arXiv.2508.05279},
  url = {https://arxiv.org/abs/2508.05279}
}
```

## License

This repository is released under the MIT License. See `LICENSE` for details.
