from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.cost_model.architecture import QWEN235, AttentionConfig
from src.cost_model.hardware import H100, HardwareConfig


class CPStrategy(ABC):
    name: str = "base"

    def __init__(
        self, cp: int, hw: HardwareConfig = H100, attn: AttentionConfig = QWEN235
    ):
        self.cp = cp
        self.hw = hw
        self.attn = attn

    @abstractmethod
    def total_time(self, batch: list[int]) -> float:
        """Total time for this strategy given a batch of sequence lengths."""
        raise NotImplementedError
