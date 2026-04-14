---
issue: 3
parents: []
eval_version: eval-v1
metric: 0.393
---

# Research Notes

## Result: Sum-of-Top-K SA achieves metric 0.393

The Erdos Minimum Overlap problem asks us to partition {1,...,2n} into equal sets A, B to minimize the normalized maximum overlap: max_k |{(a,b) : a in A, b in B, a+b=k}| / n. The theoretical lower bound is 0.381.

## Why naive approaches fail

Before finding what works, it is worth understanding what does not. Every structured partition I tested performed WORSE than random:

| Construction | Metric | Problem |
|---|---|---|
| SLP block (various sizes) | 0.51-0.98 | Creates periodic peaks in overlap |
| Lattice (golden ratio, sqrt2) | ~1.00 | Extreme concentration at specific sums |
| Quadratic residues mod p | 0.52-0.95 | Unbalanced residue classes |
| Binary XOR patterns | 0.54-0.98 | Too much algebraic structure |
| Fourier feature assignment | 0.70-0.93 | Periodic structure amplifies overlap |
| Continuous relaxation + rounding | 0.54 | Rounding destroys continuous optimum |
| **Random partition** | **0.52** | **Baseline** |

The reason is simple: any deterministic structure creates correlations in the indicator function f_A, which in Fourier space means energy concentration at specific frequencies. The convolution f_A * f_B then has peaks at sums corresponding to those frequencies. Random partitions are the best starting point precisely because they lack this structure.

## The whack-a-mole barrier at metric 0.404

Standard simulated annealing with single-element swaps converges to metric ~0.404 regardless of temperature schedule (tested T_start from 1.0 to 100.0). The reason: using max(r) as the objective creates many near-degenerate local minima. When SA reduces the peak at sum k*, another sum k' rises to nearly the same level. The SA accepts the swap (delta ~ 0), but the global structure has not improved.

This barrier appeared with:
- 500k iterations in 43 seconds
- Direct numpy fancy indexing (10x faster than np.add.at)
- Various temperature schedules
- Iterated local search with perturbation
- Targeted swaps focusing on peak contributors

## Breaking through: sum-of-top-K objective

The breakthrough came from changing the SA objective. Instead of minimizing max(r), we minimize the sum of the top K values in the overlap array r. With K=50:

**score(r) = sum of 50 largest values in r**

This forces the SA to flatten ALL peaks simultaneously, not just the tallest one. A swap that reduces one peak but raises 3 others (from 400 to 401) is now penalized, whereas the max-only objective would accept it as neutral.

The effect is dramatic:

| Objective | Metric (mean 3 seeds) |
|---|---|
| max(r) | 0.404 +/- 0.001 |
| max(r) + 0.01 * count(r >= max-2) | 0.397 +/- 0.001 |
| sum of top-50 | **0.393 +/- 0.001** |
| sum of top-75 | 0.393 +/- 0.001 |
| sum of top-100 | 0.394 +/- 0.001 |

K=50 is the sweet spot: broad enough to flatten multiple peaks, focused enough that the optimization still drives down the max.

## Detailed results

| Seed | Metric | Time |
|------|--------|------|
| 1    | 0.394  | ~44s |
| 2    | 0.393  | ~44s |
| 3    | 0.393  | ~44s |
| **Mean** | **0.393 +/- 0.001** | |

## Technical details

- **Incremental updates**: each swap changes O(n) entries in the overlap array. Using numpy fancy indexing (not np.add.at) gives ~11k iterations/second.
- **Temperature schedule**: geometric cooling from T=100 to T=0.01. Higher T compensates for the larger delta scale (sum of 50 values vs single max).
- **Partition size**: n=1000, so the search space is C(2000,1000) ~ 10^{600}.

## Prior Art and Novelty

### What is already known
- Erdos minimum overlap has theoretical lower bound ~0.381 (established in combinatorial number theory)
- SLP-lattice-anneal constructions achieve ~0.381 for large n
- SA with careful temperature scheduling is standard for combinatorial optimization

### What this orbit adds
- The sum-of-top-K SA objective is a simple but effective technique for minimax optimization. It reduces metric from 0.404 to 0.393 with no additional computational cost per iteration beyond one np.partition call.
- Systematic evaluation showing that ALL structured initial partitions perform worse than random for this problem.

### Honest positioning
This orbit applies SA with a modified objective function. The sum-of-top-K technique is known in the optimization literature (it is related to CVaR / conditional value-at-risk minimization). The contribution is demonstrating its effectiveness for the specific Erdos overlap problem and identifying the optimal K value.

## Glossary

- **SA**: Simulated Annealing
- **SLP**: Straight-Line Program (a construction method for optimal partitions)
- **Top-K objective**: SA objective equal to the sum of the K largest overlap values
- **FFT**: Fast Fourier Transform (used to compute initial overlap array via convolution)

## References

- Erdos minimum overlap problem (combinatorial number theory)
- Cilleruelo (2018) construction approach
- Conditional Value-at-Risk (CVaR) as a smooth approximation to worst-case optimization
