import matplotlib.pyplot as plt
import numpy as np

from src.agents import CounterfactualQLearningAgent, QLearningAgent
from src.core import crossproduct_factory


def main():
    """Sample efficiency comparison between Q-Learning and Counterfactual Q-Learning."""
    env = crossproduct_factory()
    ql_agent = QLearningAgent(
        env,
        epsilon=0.1,
        learning_rate=0.1,
        discount_factor=0.9,
    )
    cql_agent = CounterfactualQLearningAgent(
        env,
        epsilon=0.1,
        learning_rate=0.1,
        discount_factor=0.9,
    )

    ql_returns = ql_agent.learn(total_episodes=500)
    cql_returns = cql_agent.learn(total_episodes=500)
    smooth_ql_returns = np.convolve(ql_returns, np.ones(10) / 10, mode="valid")
    smooth_cql_returns = np.convolve(cql_returns, np.ones(10) / 10, mode="valid")

    plt.plot(smooth_ql_returns, label="Q-Learning")
    plt.plot(smooth_cql_returns, label="Counterfactual Q-Learning")
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Return (window size 10)")
    plt.title("Sample Efficiency Comparison")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
