"""Baseline: random partition of {1,...,2n} into two equal sets."""
import numpy as np


def solve(n=1000, seed=42):
    """Return set A (subset of {1,...,2n} with |A|=n) via random partition."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(2 * n) + 1  # {1, ..., 2n}
    A = set(perm[:n].tolist())
    return A
