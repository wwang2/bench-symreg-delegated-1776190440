---
issue: 3
parents: []
eval_version: eval-v1
metric: null
---

# Research Notes

## Approach: Fourier-based optimization of the Erdos Minimum Overlap partition

### Problem Setup

We partition {1,...,2n} into sets A, B with |A|=|B|=n. The overlap profile is the convolution f_A * f_B, where f_A and f_B are indicator functions. We want to minimize the L-infinity norm (max value) of this convolution, normalized by n.

In Fourier domain, convolution becomes pointwise multiplication: FFT(f_A) * FFT(f_B). The strategy is:

1. **Continuous relaxation**: Let x in [0,1]^{2n} represent the "probability" that element i is in A. Then f_B = 1 - x. Constraint: sum(x) = n.
2. **Objective**: Minimize max of IFFT(FFT(x) * FFT(1-x)) subject to constraints.
3. **Rounding**: After optimization, assign the top-n values to A.

### Known constructions

The best known approaches use lattice-based packings and simulated annealing, achieving ~0.381. We will try:
- Direct spectral optimization via scipy
- CMA-ES on the continuous relaxation
- Greedy construction informed by spectral analysis
