"""Train and evaluate the high-dimensional Double DQN experiment."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beergame.policy.double_dqn import DoubleDQNAgent
from train_high_dim_dqn import run_high_dim_dqn_experiment


if __name__ == "__main__":
    summary = run_high_dim_dqn_experiment(
        name="high_dim_double_dqn",
        agent_class=DoubleDQNAgent,
        algorithm="high_dim_double_dqn",
        seed_offset=100,
    )
    print("high_dim_double_dqn summary:", summary)
