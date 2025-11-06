from src.core.crossproduct import CrossProduct
from src.core.ground import LetterWorld
from src.core.label import LabellingFunction
from src.core.machine import RewardMachine


def crossproduct_factory():
    """Factory function to create a CrossProduct environment."""
    ground_env = LetterWorld(switch_proba=0.0)
    machine = RewardMachine()
    labelling_function = LabellingFunction()
    env = CrossProduct(ground_env, machine, labelling_function)
    return env
