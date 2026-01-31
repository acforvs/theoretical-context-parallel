from src.imbalance.base import BaseSimulator, SimulationConfig
from src.utils.estimate import approximate_flash_cost


class TheoreticalSimulator(BaseSimulator):
    def __init__(self, config: SimulationConfig):
        super().__init__(config)

    def compute_cost(self, batch: list[int]) -> float:
        # Divide by CP since Ulysses processes 1/CPth of global heads
        return approximate_flash_cost(batch) / self.config.cp
