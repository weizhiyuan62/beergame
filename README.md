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

### DQN Baseline

Run the original single-agent DQN baseline:

```bash
conda activate beergame
python -m beergame.policy.dqn
```

This trains firm `1` by default, while the other firms use random orders. It saves model checkpoints under `models/` and plots under `figures/`.

### DQN vs Double DQN Comparison

Run the reproducible comparison experiment:

```bash
conda activate beergame
python -m beergame.policy.experiment
```

This script trains and tests:

- `baseline_dqn`: standard DQN agent for firm `1`;
- `double_dqn`: Double DQN agent for firm `1`.

The non-learning firms use a base-stock heuristic policy. The script prints summary metrics including mean test reward, reward standard deviation, average inventory, average order quantity, and demand satisfaction rate.

Expected outputs:

```text
models/baseline_dqn_firm_1_final.pth
models/double_dqn_firm_1_final.pth
figures/baseline_dqn_training_rewards.png
figures/baseline_dqn_test_results.png
figures/double_dqn_training_rewards.png
figures/double_dqn_test_results.png
figures/comparison_training_curve.png
figures/comparison_test_scores.png
```

`models/` and `figures/` are ignored by git, so include them manually in the final assignment package if required.

### High-Dimensional Order-Space Experiment

Run the high-dimensional action-space comparison:

```bash
conda activate beergame
python -m beergame.policy.high_dim_experiment
```

This experiment compares:

- `high_dim_dqn`: DQN agent for firm `1` with a vector-valued order action;
- `high_dim_double_dqn`: Double DQN agent for firm `1` with the same action space.

The order action has three dimensions. Each dimension can choose from `[0, 5, 10, 15, 20]`, and only actions with total order quantity between `5` and `20` are used. The non-learning firms use a base-stock heuristic policy.

Expected outputs:

```text
models/high_dim_dqn_firm_1_episode_500.pth
models/high_dim_dqn_firm_1_episode_1000.pth
models/high_dim_dqn_firm_1_final.pth
models/high_dim_double_dqn_firm_1_episode_500.pth
models/high_dim_double_dqn_firm_1_episode_1000.pth
models/high_dim_double_dqn_firm_1_final.pth
figures/high_dim_dqn_training_rewards.png
figures/high_dim_dqn_test_results.png
figures/high_dim_double_dqn_training_rewards.png
figures/high_dim_double_dqn_test_results.png
figures/high_dim_comparison_training_curve.png
figures/high_dim_comparison_test_scores.png
```
