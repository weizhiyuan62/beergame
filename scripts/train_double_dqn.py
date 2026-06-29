"""Train and evaluate the Double DQN experiment."""

from pathlib import Path
import sys

import hydra

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beergame.policy.double_dqn import DoubleDQNAgent
from train_dqn import run_dqn_experiment
from train_config import hydra_config_to_dict, runtime_start


@hydra.main(version_base=None, config_path="../cfg", config_name="double_dqn")
def main(cfg):
    cfg = hydra_config_to_dict(cfg)
    runtime = runtime_start()
    summary = run_dqn_experiment(
        agent_class=DoubleDQNAgent,
        config=cfg,
        runtime=runtime,
    )
    print("double_dqn summary:", summary)


if __name__ == "__main__":
    main()
