#!/usr/bin/env python3
"""Evaluator for the Erdos Minimum Overlap problem.

Loads a solution module that provides `solve(n, seed)` returning a set A
(subset of {1,...,2n} with |A|=n). Computes the normalized maximum overlap:
    max_k |{(a,b) : a in A, b in B, a+b=k}| / n
where B = {1,...,2n} \\ A.

Uses FFT-based convolution for efficiency.
"""
import argparse
import importlib.util
import sys
import numpy as np


def load_solution(path):
    spec = importlib.util.spec_from_file_location("solution", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_overlap(A, n):
    """Compute normalized max overlap for partition A, B={1..2n}\\A.

    Returns max_k |{(a,b): a in A, b in B, a+b=k}| / n.
    """
    full = set(range(1, 2 * n + 1))
    A = set(A)
    B = full - A

    assert len(A) == n, f"Expected |A|={n}, got {len(A)}"
    assert len(B) == n, f"Expected |B|={n}, got {len(B)}"
    assert A | B == full, "A and B must partition {1,...,2n}"

    # Indicator arrays (1-indexed, so size 2n+1)
    f_A = np.zeros(2 * n + 1)
    f_B = np.zeros(2 * n + 1)
    for a in A:
        f_A[a] = 1.0
    for b in B:
        f_B[b] = 1.0

    # Convolution: (f_A * f_B)[k] = |{(a,b): a in A, b in B, a+b=k}|
    conv = np.convolve(f_A, f_B)
    max_count = float(np.max(conv))
    return max_count / n


def evaluate(solution_path, seed=42):
    module = load_solution(solution_path)
    n = 1000  # Problem size
    A = module.solve(n=n, seed=seed)
    metric = compute_overlap(A, n)
    return metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Erdos Minimum Overlap evaluator")
    parser.add_argument("--solution", required=True, help="Path to solution.py")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    metric = evaluate(args.solution, args.seed)
    print(f"METRIC={metric:.6f}")
