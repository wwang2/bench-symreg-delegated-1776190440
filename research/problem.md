# Erdos Minimum Overlap

## Question

Find partitions of {1, ..., 2n} into two sets A, B with |A| = |B| = n that minimize the maximum overlap count max_k |{(a, b) : a in A, b in B, a + b = k}|, normalized by n.

## Background

The Erdos minimum overlap problem asks: given a partition of {1, ..., 2n} into two equal sets A and B, what is the minimum possible value of max_k r(k), where r(k) counts the number of pairs (a, b) with a in A, b in B, and a + b = k?

The normalized overlap (dividing by n) converges to a constant as n grows. The established theoretical lower bound is approximately 0.380926. The best known construction achieves ~0.38087 via SLP-lattice-anneal strategies.

## Known Results

- Random partitions: ~0.49 normalized overlap
- Lattice packing: ~0.403
- SLP (Straight-Line Program) block 1024: ~0.391
- SLP seeded from lattice: ~0.382
- SLP-lattice-anneal (current best): ~0.38087
- Theoretical lower bound: 0.380926

## Success Criteria

Metric: normalized maximum overlap (minimize). Target: 0.01.
Budget: 3 orbits maximum.

## References

- Erdos minimum overlap problem (combinatorial number theory)
- Cilleruelo (2018) construction approach
