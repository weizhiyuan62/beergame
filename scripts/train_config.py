"""Small config and runtime helpers for training scripts."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


@dataclass
class EnvConfig:
    num_firms: int = 3
    p: list[int] = field(default_factory=lambda: [10, 9, 8])
    h: float = 0.5
    c: float = 2
    initial_inventory: int = 100
    poisson_lambda: int = 10
    max_steps: int = 100


@dataclass
class TrainingConfig:
    name: str = "baseline_dqn"
    algorithm: str = "dqn"
    result_root: str = "results"
    firm_id: int = 1
    action_size: int = 20
    num_episodes: int = 1000
    test_episodes: int = 20
    seed: int = 42
    seed_offset: int = 0
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay: float = 0.995


@dataclass
class DQNConfig:
    buffer_size: int = 10000
    batch_size: int = 64
    gamma: float = 0.99
    learning_rate: float = 1e-3
    tau: float = 1e-3
    update_every: int = 4


@dataclass
class PPOConfig:
    learning_rate: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.15
    update_epochs: int = 4
    minibatch_size: int = 64
    entropy_coef: float = 0.03
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    reward_scale: float = 0.01


@dataclass
class HighDimConfig:
    order_dim: int = 3
    order_levels: list[int] = field(default_factory=lambda: [0, 5, 10, 15, 20])
    order_cost_multipliers: list[float] = field(default_factory=lambda: [1.0, 1.1, 1.25])
    min_total_order: int = 5
    max_total_order: int = 20


@dataclass
class TrainConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    high_dim: HighDimConfig = field(default_factory=HighDimConfig)


def hydra_config_to_dict(config):
    """Convert Hydra/OmegaConf config objects into plain JSON-serializable dicts."""
    merged = OmegaConf.merge(OmegaConf.structured(TrainConfig), config)
    return OmegaConf.to_container(merged, resolve=True)


def runtime_start():
    hydra_output_dir = None
    if HydraConfig.initialized():
        hydra_output_dir = HydraConfig.get().runtime.output_dir

    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "_start_perf": time.perf_counter(),
        "command": sys.argv,
        "hydra_output_dir": hydra_output_dir,
    }


def runtime_finish(runtime):
    finished = datetime.now(timezone.utc).isoformat()
    duration = time.perf_counter() - runtime["_start_perf"]
    return {
        "started_at": runtime["started_at"],
        "finished_at": finished,
        "duration_seconds": duration,
        "command": runtime["command"],
        "hydra_output_dir": runtime.get("hydra_output_dir"),
    }


def as_plain_config(config):
    """Return a JSON-serializable config copy without private helper keys."""
    return {k: v for k, v in config.items() if not k.startswith("_")}


def default_config_dict():
    return asdict(TrainConfig())


def result_dir_for_run(config, name, result_root):
    """Use Hydra's run directory when available; otherwise create a timestamped result dir."""
    if HydraConfig.initialized():
        return Path(HydraConfig.get().runtime.output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(result_root) / name / timestamp
