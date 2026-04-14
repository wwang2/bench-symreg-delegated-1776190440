---
issue: 2
parents: []
eval_version: eval-v1
metric: 0.391
---

# Research Notes

## Summary

This orbit explores simulated annealing with pairwise swaps for the Erdos Minimum Overlap problem. The critical finding is that a **L2 surrogate objective** dramatically outperforms the naive hard-max objective, achieving metric=0.391 versus ~0.40 for hard-max SA.

## Why the interleaved start fails

The original hypothesis was that A=odds, B=evens would provide a good structured starting point. This is wrong. The interleaved partition has metric=1.0 because all 1000 pairs (a,b) with a odd, b even satisfy a+b=2001, giving r(2001)=1000. This is the worst possible partition, not the best.

Random partitions give metric ~0.50, which is a much better starting point.

## The L2 surrogate insight

Standard SA with the hard-max objective (minimize max_k r(k)) gets stuck at metric ~0.40 because most random swaps do not change the maximum. The acceptance rate for improving swaps becomes vanishingly small near a local minimum.

The L2 surrogate replaces the hard max with a softer penalty:

    cost = sum(max(0, r[k] - threshold)^2)

This has several advantages:
1. It penalizes ALL peaks above the threshold, not just the single maximum.
2. Most swaps DO change the L2 cost, providing useful gradient signal.
3. As the threshold decreases over the annealing schedule, the overlap profile flattens progressively.
4. The hard max naturally decreases as the profile flattens.

The threshold schedule starts at 90% of the initial max and decays toward the theoretical lower bound (0.38 * n) over the course of the run.

## Implementation

The inner SA loop is compiled to C via ctypes for maximum throughput (~120K iterations/sec with L2 cost tracking). Each iteration:
1. Pick random elements a in A, b in B
2. Apply the swap to the overlap array r[] (O(n) work)
3. Track the L2 cost change incrementally (no need for full recomputation)
4. Accept/reject based on the L2 cost change and temperature
5. Revert if rejected

The L2 phase runs for 82% of the time budget. The final 18% switches to hard-max SA for fine polishing.

## Results

| Seed | Metric | Time |
|------|--------|------|
| 1    | 0.391  | 42s  |
| 2    | 0.391  | 42s  |
| 3    | 0.391  | 42s  |
| **Mean** | **0.391 +/- 0.000** | |

## Comparison with baselines

- Random partition: ~0.50
- Hard-max SA (numpy): ~0.40
- Hard-max SA (C-compiled): ~0.40
- **L2-surrogate SA (C-compiled): 0.391**
- Lattice packing (literature): ~0.403
- SLP block 1024 (literature): ~0.391
- SLP-lattice-anneal (literature best): ~0.381
- Theoretical lower bound: 0.3809

Our result matches the SLP block 1024 result from the literature, achieved through a simpler and more general method.

## What didn't work

1. **Interleaved start (A=odds)**: metric=1.0, the worst possible.
2. **Lattice/sequence constructions** (Beatty, Thue-Morse, QR, Rudin-Shapiro): all worse than random (0.52-0.99).
3. **Greedy construction** (assign elements one-by-one): gives ~0.50, same as random.
4. **Multi-restart with short runs**: 0.393, worse than single long run (0.391).
5. **Targeted swaps** (focus on peak contributors): marginal improvement over random swaps.
6. **Reheat cycles**: no improvement over standard exponential cooling.

## Prior Art & Novelty

### What is already known
- The Erdos minimum overlap problem has a theoretical lower bound of 0.3809 (Cilleruelo, 2018).
- SLP-based constructions with lattice seeding achieve ~0.381 for large n.
- Simulated annealing with hard-max objective reaches ~0.40 for n=1000.

### What this orbit adds
- The L2 surrogate objective is a practical technique that improves SA from ~0.40 to ~0.391 without requiring specialized construction methods.
- The approach is simple to implement and generalizable to other min-max optimization problems.

### Honest positioning
This is an application of a known technique (surrogate objectives for SA) to a specific problem. The L2 surrogate is not novel in the optimization literature, but its effectiveness for the Erdos overlap problem appears not to have been documented. The result (0.391) matches but does not beat the SLP block construction, and is still 2.6% above the theoretical lower bound.

## Glossary

- **SA**: Simulated Annealing
- **SLP**: Straight-Line Program, a recursive block construction method
- **L2 surrogate**: Using sum of squared excesses above a threshold instead of the hard maximum as the optimization objective
- **QR**: Quadratic Residue

## References

- Erdos minimum overlap problem (combinatorial number theory)
- Cilleruelo (2018) construction approach, lower bound ~0.3809
