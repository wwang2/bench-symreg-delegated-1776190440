---
issue: 4
parents: []
eval_version: eval-v1
metric: 0.3893
---

# Research Notes

## Result: 0.389 normalized overlap via penalty-guided simulated annealing

The Erdos minimum overlap problem partitions {1,...,2n} into equal sets A, B and minimizes max_k r(k)/n where r(k) counts pairs (a,b) with a in A, b in B, a+b=k. We achieve 0.389, beating the lattice packing benchmark (0.403) and matching the SLP block-1024 result (0.391).

## Why standard SA gets stuck at 0.40

The overlap r(k) = sum of indicator convolution at position k. A single swap (a in A <-> b in B) modifies r at O(n) positions. The max overlap typically occurs at 5-10 positions near k=n+1. Random single swaps rarely improve all of these simultaneously, so standard SA converges to a local optimum around 0.40.

The fundamental issue: the objective max(conv) is a *flat* landscape near convergence. When the max doesn't change (the typical case), the SA has no information about which direction to move.

## The penalty-based objective

We replace the bare max objective with a soft penalty:

  obj(A) = max_k r(k) + lambda * sum_k max(0, r(k) - threshold)^2

This penalizes all overlap values above a sliding threshold, not just the maximum. The key insight: even when a swap doesn't change the max, it can improve the penalty by reducing near-max values. Over many iterations, this systematic flattening of the overlap profile eventually pushes the actual maximum down.

### Adaptive schedule

The penalty parameters adapt over the SA run:
- **Margin** (threshold = best_max - margin): starts at 20, ends at 2
- **Lambda** (penalty weight): starts at 0.005, ends at 0.1

Early on, the wide margin and low weight encourage broad exploration. Late in the run, the narrow margin and high weight focus effort on flattening the profile near the maximum.

## Implementation details

1. **Initial construction**: Quadratic Residue (QR) partition with prime p close to 2n, giving a near-balanced starting partition with metric ~0.50.

2. **SA inner loop**: In-place convolution updates in O(n) per swap. The delta formula for swapping a (in A) with b (in B) modifies conv via 4 shifted array slice operations plus 3 point corrections.

3. **Throughput**: ~25K iterations/second at n=1000, giving ~1M iterations in 44 seconds.

4. **Batched RNG**: Pre-generate random indices and uniforms in batches of 2000 to reduce per-iteration overhead.

## Evolution of results

| Iteration | Approach | Seed 1 | Seed 2 | Seed 3 | Mean |
|-----------|----------|--------|--------|--------|------|
| 1 | Greedy element-by-element | 0.501 | - | - | ~0.50 |
| 2 | SA from QR init, max objective | 0.404 | 0.404 | 0.405 | 0.404 |
| 3 | SA with in-place updates | 0.402 | 0.402 | 0.401 | 0.402 |
| 4 | SA with penalty (static) | 0.394 | 0.392 | 0.392 | 0.393 |
| 5 | SA with penalty (adaptive) | 0.390 | 0.390 | 0.390 | 0.390 |
| 6 | SA with tuned adaptive schedule | 0.389 | 0.390 | 0.389 | **0.389** |

## What did not work

- **Pure greedy construction**: element-by-element greedy produces random-level results (0.50)
- **Modular/block patterns**: periodic constructions give 0.60-1.00 before SA, worse than random
- **Kick-restart SA**: random perturbation + restart wastes time vs continuous SA
- **Targeted swaps**: Python loops to find hot-sum elements are too slow
- **L1.5 vs L2 penalty**: no significant difference
- **Lazy penalty evaluation**: computing penalty only on ties doesn't help enough

## Prior Art and Novelty

### What is already known
- Lattice packing achieves ~0.403 (classical result)
- SLP block-1024 achieves ~0.391 (Cilleruelo-style construction)
- SLP-lattice-anneal achieves ~0.381 (current best)
- Theoretical lower bound is 0.380926

### What this orbit adds
- Demonstrates that penalty-based SA can match SLP performance (~0.389) without requiring the complex SLP construction
- The adaptive penalty schedule (margin + lambda annealing) is a practical technique for minimax optimization via SA
- This is an application of known SA + penalty methods -- no novelty claim beyond the specific parameter tuning

### Honest positioning
Our result of 0.389 sits between lattice packing (0.403) and SLP-lattice-anneal (0.381). The penalty-based SA is simpler to implement than SLP but reaches diminishing returns around 0.389 due to the single-swap move limitation. To approach the theoretical lower bound of 0.381, multi-element moves or block-structured approaches (like SLP) appear necessary.

## References

- Erdos minimum overlap problem (combinatorial number theory)
- Cilleruelo (2018) construction approach for SLP partitions
