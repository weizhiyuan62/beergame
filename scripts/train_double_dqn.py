"""Run only the Double DQN experiment and save outputs to results/."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beergame.policy.runner import run_double_dqn


if __name__ == "__main__":
    summary = run_double_dqn()
    print("double_dqn summary:", summary)
