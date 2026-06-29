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
  agents/      # learning algorithms: DQN, Double DQN, PPO, high-dimensional helpers
  heuristic/   # non-learning baseline policies
  runner.py    # shared train/test/result-saving workflow used by scripts/
```

### DQN Baseline

Run the original single-agent DQN baseline:

```bash
conda activate beergame
python scripts/train_dqn.py
```

This trains firm `1` by default, while the other firms use base-stock orders. It saves raw training and test results under `results/baseline_dqn/`.

### Double DQN

Run the Double DQN improved algorithm:

```bash
conda activate beergame
python scripts/train_double_dqn.py
```

This saves raw training and test results under `results/double_dqn/`.

### PPO Improved-Algorithm Experiment

Run PPO as a separate improved-algorithm experiment:

```bash
conda activate beergame
python scripts/train_ppo.py
```

This saves raw training and test results under `results/ppo/`.

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
    model.pth
    training_scores.npy
    test_scores.npy
    test_history.npz
    summary.json
  double_dqn/
  ppo/
  summary/baseline_dqn_double_dqn_ppo/
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
```

This experiment compares:

- `high_dim_dqn`: DQN agent for firm `1` with a vector-valued order action;
- `high_dim_double_dqn`: Double DQN agent for firm `1` with the same action space.

The order action has three dimensions. Each dimension can choose from `[0, 5, 10, 15, 20]`, and only actions with total order quantity between `5` and `20` are used. The non-learning firms use a base-stock heuristic policy.

To aggregate high-dimensional results, edit `scripts/plot_results.py`:

```python
target_list = ["high_dim_dqn", "high_dim_double_dqn"]
```
