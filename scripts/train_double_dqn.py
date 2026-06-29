"""Train and evaluate the Double DQN experiment."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beergame.policy.double_dqn import DoubleDQNAgent
from train_dqn import run_dqn_experiment


if __name__ == "__main__":
    summary = run_dqn_experiment(
        name="double_dqn",
        agent_class=DoubleDQNAgent,
        algorithm="double_dqn",
        seed_offset=100,
    )
    print("double_dqn summary:", summary)
