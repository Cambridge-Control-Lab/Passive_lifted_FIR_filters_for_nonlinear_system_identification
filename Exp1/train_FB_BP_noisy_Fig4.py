from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER_MODULE = "Exp1.Core_code.Exp1_NFIR_training_code"


FB_BP = [
    { # FB-BP noise 
        "is_passive": "1",
        "noise_on": "1",
        "hidden_dims": "4,4",
        "active_dims": "0,1",
        "delay_steps_by_dim": "{}", "exp_id": "0", # exp_id shows delaysteps in run name, 0 means no delay. 2 means 2 delays 
        "m_fir": "50",
        "n_branch_all_cases": "10",
        "max_iteration": "5",
        "step2_l2reg": "1.0",
        "last_bptt_on": "1",
    }
]
experiments = (FB_BP)
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
        "--last-bptt-on", exp_cfg.get("last_bptt_on", "1"),
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
