def approximate_flash_cost(batch: list[int]) -> float:
    """Approximate flash attention cost as sum of squared sequence lengths."""
    return float(sum(s * s for s in batch))
