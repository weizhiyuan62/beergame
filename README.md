# BeergameProject-2026PKUSpring

- Beer Game course project for supply chain simulation, reinforcement learning experiments, and final report writing.

## Env
- env set:(you need to have conda first)
```bash
cd beergame
conda create -n beergame python=3.10.16 -y
conda activate beergame
pip install -e .
```


## Experiments

All experiment outputs are written under `results/`, which is ignored by git.

Policy code is organized as:

```text
src/beergame/policy/
  dqn.py             # DQN agent
  double_dqn.py      # Double DQN agent
  ppo.py             # PPO agent
  high_dim.py        # high-dimensional action-space definition
  base_stock.py      # base-stock heuristic policy
  random_policy.py   # random heuristic policy
scripts/
  train_*.py         # experiment-specific train/test/result-saving workflows
  plot_results.py    # aggregate plotting from results/
cfg/
  *.yaml             # experiment configs used by train scripts
```

### Hydra Configs

Each training script is managed by Hydra. It loads its matching YAML config by default and supports command-line overrides:

```bash
python scripts/train_dqn.py training.num_episodes=2000 dqn.learning_rate=0.0005
python scripts/train_ppo.py ppo.entropy_coef=0.05 training.result_root=results/ppo_entropy
```

Use `--config-name` if you add another YAML file under `cfg/`.

### DQN Baseline

Run the original single-agent DQN baseline:

```bash
conda activate beergame
python scripts/train_dqn.py
```

This trains firm `1` by default, while the other firms use base-stock orders. It saves raw training and test results under `results/baseline_dqn/<run_time>/`.
The default scalar-action setting trains DQN-style agents for 1000 episodes and evaluates on 20 test episodes.

### Double DQN

Run the Double DQN improved algorithm:

```bash
conda activate beergame
python scripts/train_double_dqn.py
```

This saves raw training and test results under `results/double_dqn/<run_time>/`.

### PPO Improved-Algorithm Experiment

Run PPO as a separate improved-algorithm experiment:

```bash
conda activate beergame
python scripts/train_ppo.py
```

This saves raw training and test results under `results/ppo/<run_time>/`.
PPO uses a longer default horizon of 1500 training episodes, reward scaling for policy updates, and stronger entropy regularization to reduce premature low-order policies.

### Aggregate Plotting

Edit `target_list` in `scripts/plot_results.py` to choose which experiments are included, for example:

```python
target_list = ["baseline_dqn", "double_dqn", "ppo"]
```

Then generate comparison figures:

```bash
conda activate beergame
python scripts/plot_results.py
```

Expected output structure:

```text
results/
  baseline_dqn/
    20260629_153012/
      .hydra/
        config.yaml
        hydra.yaml
        overrides.yaml
      train_dqn.log
      model.pth
      training_scores.npy
      test_scores.npy
      test_history.npz
      summary.json    # metrics, full Hydra config, runtime, and command
  double_dqn/
  ppo/
  summary/20260629_154455/
    comparison_training_curve.png
    comparison_test_scores.png
    comparison_behavior.png
```

### High-Dimensional Order-Space Experiment

Run the high-dimensional action-space comparison:

```bash
conda activate beergame
python scripts/train_high_dim_dqn.py
python scripts/train_high_dim_double_dqn.py
python scripts/train_high_dim_ppo.py
```

This experiment compares:

- `high_dim_dqn`: DQN agent for firm `1` with a vector-valued order action;
- `high_dim_double_dqn`: Double DQN agent for firm `1` with the same action space;
- `high_dim_ppo`: PPO agent for firm `1` over the same enumerated vector action space.

The order action has three dimensions. Each dimension can choose from `[0, 5, 10, 15, 20]`, and only actions with total order quantity between `5` and `20` are used. PPO treats the enumerated vector actions as a categorical action space, then maps selected action indices back to order vectors before stepping the environment. The non-learning firms use a base-stock heuristic policy.
The high-dimensional DQN scripts train for 1500 episodes by default, while high-dimensional PPO uses its own config under `cfg/high_dim_ppo.yaml`; all evaluate on 20 test episodes.

To aggregate high-dimensional results, edit `scripts/plot_results.py`:

```python
target_list = ["high_dim_dqn", "high_dim_double_dqn", "high_dim_ppo"]
```
