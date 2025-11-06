class RewardMachine:
    """Reward machine for LetterWorld task.

    Agent is required to observe the following sequence of
    letters: A -> C -> A -> C
    """

    def __init__(self):
        """Initialise the reward machine."""
        self.delta_u = self._delta_u()
        self.delta_r = self._delta_r()
        self.u_0 = 0  # Initial machine state
        self.U = (0, 1, 2, 3)  # Set of machine states
        self.F = (4,)  # Set of terminal states

    def _delta_u(self) -> dict[tuple[int, str], int]:
        """State-transition function of the reward machine."""
        return {
            # Transitions for machine state u0
            (0, "A"): 1,
            (0, "B"): 0,
            (0, "C"): 0,
            (0, ""): 0,
            # Transitions for machine state u1
            (1, "A"): 1,
            (1, "B"): 1,
            (1, "C"): 2,
            (1, ""): 1,
            # Transitions for machine state u2
            (2, "A"): 3,
            (2, "B"): 2,
            (2, "C"): 2,
            (2, ""): 2,
            # Transitions for machine state u3
            (3, "A"): 3,
            (3, "B"): 3,
            (3, "C"): 4,
            (3, ""): 3,
        }

    def _delta_r(self) -> dict[tuple[int, str], float]:
        return {
            # Reward for transitions from machine state u0
            (0, "A"): 0.0,
            (0, "B"): 0.0,
            (0, "C"): 0.0,
            (0, ""): 0.0,
            # Reward for transitions from machine state u1
            (1, "A"): 0.0,
            (1, "B"): 0.0,
            (1, "C"): 0.0,
            (1, ""): 0.0,
            # Reward for transitions from machine state u2
            (2, "A"): 0.0,
            (2, "B"): 0.0,
            (2, "C"): 0.0,
            (2, ""): 0.0,
            # Reward for transitions from machine state u3
            (3, "A"): 0.0,
            (3, "B"): 0.0,
            (3, "C"): 1.0,
            (3, ""): 0.0,
        }

    def transition(self, u: int, props: str) -> tuple[int, float]:
        """Get the next machine state and reward given current state and observation.

        Args:
            u: Machine state before transitioning.
            props: Observed proposition/event.

        Returns:
            tuple[int, float]: Next machine state and reward.
        """
        try:
            u_next = self.delta_u[(u, props)]
            r = self.delta_r[(u, props)]
            return u_next, r
        except KeyError as e:
            raise ValueError(
                f"Invalid transition from state {u} with observation '{props}'"
            ) from e
