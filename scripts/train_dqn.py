"""Run only the baseline DQN experiment and save outputs to results/."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beergame.policy.runner import run_baseline_dqn


if __name__ == "__main__":
    summary = run_baseline_dqn()
    print("baseline_dqn summary:", summary)
