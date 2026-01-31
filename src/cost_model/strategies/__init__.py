from src.cost_model.strategies.base import CPStrategy
from src.cost_model.strategies.ring import RingAttention
from src.cost_model.strategies.ulysses import UlyssesAttention
from src.cost_model.strategies.zigzag import ZigZagAttention

__all__ = [
    "CPStrategy",
    "UlyssesAttention",
    "RingAttention",
    "ZigZagAttention",
]

STRATEGIES = {
    "ulysses": UlyssesAttention,
    "ring": RingAttention,
    "zigzag": ZigZagAttention,
}
