import numpy as np

from beergame.env import Env


def run_random_policy_example(num_episodes=5):
    num_firms = 3
    p = [10, 9, 8]
    h = 0.5
    c = 2
    initial_inventory = 100
    poisson_lambda = 10
    max_steps = 100

    env = Env(num_firms, p, h, c, initial_inventory, poisson_lambda, max_steps)

    for episode in range(num_episodes):
        env.reset()
        total_rewards = np.zeros((num_firms, 1))
        done = False

        while not done:
            actions = np.random.randint(1, 21, size=(num_firms, 1))
            _, rewards, done = env.step(actions)
            total_rewards += rewards
            print(
                f"Episode {episode + 1}, Step {env.current_step}, "
                f"Rewards: {rewards.T}, Total Rewards: {total_rewards.T}"
            )


if __name__ == "__main__":
    run_random_policy_example()
