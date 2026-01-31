from dataclasses import dataclass


@dataclass
class AttentionConfig:
    num_heads: int = 64
    head_dim: int = 128

    num_kv_heads: int = 8
    """Use GQA."""


QWEN235 = AttentionConfig()
