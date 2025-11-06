import gymnasium as gym
import numpy as np

from src.core.label import LabellingFunction
from src.core.machine import RewardMachine


class CrossProduct(gym.Env):
    """Cross-product MDP evironment."""

    def __init__(
        self,
        ground_env: gym.Env,
        machine: RewardMachine,
        labelling_function: LabellingFunction,
        max_steps: int = 100,
    ):
        """Initialise the cross-product MDP environment.

        Args:
            ground_env: The ground MDP environment.
            machine: The reward machine.
            labelling_function: The labelling function.
            max_steps: Maximum number of steps per episode.
        """
        super().__init__()
        self.ground_env = ground_env
        self.machine = machine
        self.labelling_function = labelling_function

        self.action_space = ground_env.action_space
        self.observation_space = gym.spaces.Box(
            low=0,
            high=100,
            shape=(self.ground_env.observation_space.shape[0] + 1,),
            dtype=np.int32,
        )

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment."""
        super().reset(seed=seed, options=options)

        self.steps = 0
        self.u = self.machine.u_0

        # Store ground observation for CFEG
        self._ground_obs, _ = self.ground_env.reset()
        self._ground_obs_next = self._ground_obs

        obs = self._get_obs(self._ground_obs, self.u)
        return obs, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment."""
        self.steps += 1
        self._ground_obs = self._ground_obs_next

        self._ground_obs_next, _, _, _, _ = self.ground_env.step(action)
        self._props = self.labelling_function(
            self._ground_obs, action, self._ground_obs_next
        )
        self.u, reward = self.machine.transition(self.u, self._props)

        terminated = self.u in self.machine.F
        truncated = self.steps >= 100

        obs = self._get_obs(self._ground_obs_next, self.u)
        return obs, reward, terminated, truncated, {}

    def generate_counterfactual_experience(
        self, ground_obs: np.ndarray, action: int, next_ground_obs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate counterfactual experiences."""
        (
            obs_buffer,
            action_buffer,
            obs_next_buffer,
            reward_buffer,
            done_buffer,
            info_buffer,
        ) = ([] for _ in range(6))

        props = self.labelling_function(ground_obs, action, next_ground_obs)

        for u_i in self.machine.U:
            try:
                u_j, r_j = self.machine.transition(u_i, props)
            except ValueError:
                # Transition is no-op, therefore skip.
                continue

            obs_buffer.append(self._get_obs(ground_obs, u_i))
            action_buffer.append(action)
            obs_next_buffer.append(self._get_obs(next_ground_obs, u_j))
            reward_buffer.append(r_j)
            done_buffer.append(u_j in self.machine.F)
            info_buffer.append({})

        return (
            np.array(obs_buffer),
            np.array(action_buffer),
            np.array(obs_next_buffer),
            np.array(reward_buffer),
            np.array(done_buffer),
            np.array(info_buffer),
        )

    def _get_obs(self, ground_obs: np.ndarray, u: int) -> np.ndarray:
        """Get the cross-product observation.

        Args:
            ground_obs: The ground observation.
            u: The current machine state.

        Returns:
            The cross-product observation which is the concatenation of the ground
            observation and the machine state.
        """
        return np.concatenate((ground_obs, np.array([u], dtype=np.int32)))

    def to_ground_obs(self, obs: np.ndarray) -> np.ndarray:
        """Convert cross-product observation to ground observation.

        Args:
            obs: The cross-product observation.

        Returns:
            The ground observation.
        """
        return obs[:-1]
