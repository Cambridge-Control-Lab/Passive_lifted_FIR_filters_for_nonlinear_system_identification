from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent


RUNNER_MODULE = "Exp2.Core_code.Exp2_NFIR"
baseline_cfg = { # dimensions: scalar dict, baseline command-line parameter set.
    "is_passive": "1",
    "noise_on": "0",
    "hidden_dims": "8,8",
    "active_dims": "0,1",
    "delay_steps_by_dim": {"0": 100, "1": 100},
    "exp_id": "100_final", # exp_id shows delaysteps in run name, 0 means no delay. 2 means 2 delays
    "m_fir": "500",
    "n_branch_all_cases": "5",
    "max_iteration": "4",
    "step2_l2reg": "20.0",
    "step1_weight_decay": "5",
    "enable_parallel_fir": "1",
    "last_bptt_on": "0",
    "step1_epoch_count": "20",
    "zero_cost_first_n": "50",
}


def make_case(base_cfg, **updates):
    cfg = deepcopy(base_cfg) # dimensions: scalar dict, copied experiment config.
    for key, value in updates.items():
        if isinstance(value, dict):
            cfg[key] = json.dumps(value)
        else:
            cfg[key] = str(value)
    if isinstance(cfg["delay_steps_by_dim"], dict):
        cfg["delay_steps_by_dim"] = json.dumps(cfg["delay_steps_by_dim"])
    return cfg

best_case = [ 
    make_case(baseline_cfg, hidden_dims="8,8", n_branch_all_cases=3, delay_steps_by_dim={"0": 150, "1": 150},  exp_id="150_final"),
]

experiments = (best_case)

results = []

for i_exp, exp_cfg in enumerate(experiments, start=1):
    exp_id = exp_cfg["exp_id"]
    print("")
    print("=" * 80)
    print(f"Starting experiment {i_exp} / {len(experiments)}: {exp_id}")
    print("=" * 80)

    cmd = [
        sys.executable,
        "-m",
        RUNNER_MODULE,
        "--exp-id", exp_cfg["exp_id"],
        "--hidden-dims", exp_cfg["hidden_dims"],
        "--active-dims", exp_cfg["active_dims"],
        "--delay-steps-by-dim", exp_cfg["delay_steps_by_dim"],
        "--m-fir", exp_cfg["m_fir"],
        "--n-branch-all-cases", exp_cfg["n_branch_all_cases"],
        "--noise-on", exp_cfg["noise_on"],
        "--is-passive", exp_cfg["is_passive"],
        "--max-iteration", exp_cfg["max_iteration"],
        "--step2-l2reg", exp_cfg["step2_l2reg"],
        "--step1-weight-decay", exp_cfg.get("step1_weight_decay", "1e-5"),
        "--enable-parallel-fir", exp_cfg.get("enable_parallel_fir", "1"),
        "--last-bptt-on", exp_cfg.get("last_bptt_on", "1"),
        "--step1-epoch-count", exp_cfg.get("step1_epoch_count", "50"),
        "--zero-cost-first-n", exp_cfg.get("zero_cost_first_n", "0"),
    ]
    result = subprocess.run(cmd, cwd=ROOT)
    return_code = int(result.returncode)
    results.append((exp_id, return_code))

    if return_code == 0:
        print(f"Finished OK: {exp_id}")
    else:
        print(f"FAILED with return code {return_code}: {exp_id}")
        print("Continue to next experiment.")

print("")
print("=" * 80)
print("Final summary")
print("=" * 80)

any_failed = False
for exp_id, return_code in results:
    if return_code == 0:
        print(f"OK:     {exp_id}")
    else:
        any_failed = True
        print(f"FAILED: {exp_id}, return code = {return_code}")

if any_failed:
    sys.exit(1)

sys.exit(0)
