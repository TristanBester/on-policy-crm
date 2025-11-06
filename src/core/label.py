import numpy as np


class LabellingFunction:
    """Simplified labelling function.

    The labelling function should support the subset of propositions/events
    which are true in the environment. Here we have simplified it to return
    at most one proposition/event.

    Propositions/events supported:
        - "A": The agent sees symbol A.
        - "B": The agent sees symbol B.
        - "C": The agent sees symbol C.
        - "": No propositions/events are true.

    """

    A_B_POSITION = np.array([1, 1])
    C_POSITION = np.array([1, 5])

    def __call__(self, obs: np.ndarray, action: int, next_obs: np.ndarray) -> str:
        """Simplified labelling function for LetterWorld environment.

        Returns:
            The proposition/event that is true in the given transition. Return empty
            string if no events are true.
        """
        del obs, action

        if next_obs[0] == 0 and np.array_equal(next_obs[1:], self.A_B_POSITION):
            return "A"
        if next_obs[0] == 1 and np.array_equal(next_obs[1:], self.A_B_POSITION):
            return "B"
        if np.array_equal(next_obs[1:], self.C_POSITION):
            return "C"
        return ""
