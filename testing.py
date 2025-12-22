# %%
import json
from pathlib import Path

BASE = Path("outputs/gph")

# Expected steps by gamma
EXPECTED_STEPS = {
    0.75: 5000,
    1.0: 10000,
    1.5: 27000,
}

# %%
# Check for incomplete runs and missing files

incomplete = []
missing_history = []
complete = []

for run_dir in sorted(BASE.iterdir()):
    if not run_dir.is_dir():
        continue

    config_path = run_dir / "config.yaml"
    history_path = run_dir / "history.json"

    # Missing history = killed before any save
    if not history_path.exists():
        missing_history.append(run_dir.name)
        continue

    # Check step count
    history = json.loads(history_path.read_text())
    last_step = history["step"][-1]

    # Extract gamma from dirname
    gamma = float(run_dir.name.split("_g")[1].split("_")[0])
    expected = EXPECTED_STEPS.get(gamma, 0) - 1  # 0-indexed

    if last_step < expected:
        incomplete.append((run_dir.name, last_step, expected, gamma))
    else:
        complete.append(run_dir.name)

# %%
# Summary

print(f"Complete: {len(complete)}")
print(f"Incomplete: {len(incomplete)}")
print(f"Missing history: {len(missing_history)}")

# %%
# Incomplete details (grouped by gamma)

if incomplete:
    print("\nIncomplete runs:")
    for gamma in [0.75, 1.0, 1.5]:
        gamma_runs = [(n, l, e) for n, l, e, g in incomplete if g == gamma]
        if gamma_runs:
            print(f"\n  γ={gamma} ({len(gamma_runs)} runs):")
            for name, last, expected in gamma_runs[:5]:  # show first 5
                print(f"    {name}: {last}/{expected} steps")
            if len(gamma_runs) > 5:
                print(f"    ... and {len(gamma_runs) - 5} more")

# %%
# Missing history details

if missing_history:
    print("\nMissing history.json:")
    for name in missing_history[:10]:
        print(f"  {name}")
    if len(missing_history) > 10:
        print(f"  ... and {len(missing_history) - 10} more")
