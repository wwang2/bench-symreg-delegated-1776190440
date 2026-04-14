---
issue: 4
parents: []
eval_version: eval-v1
metric: null
---

# Research Notes

## Approach: Greedy Partition with Local Search Refinement

The Erdos minimum overlap problem asks us to partition {1,...,2n} into two equal sets A and B such that the maximum convolution count max_k |{(a,b): a in A, b in B, a+b=k}| / n is minimized.

### Why greedy might work

The overlap profile r(k) = |{(a,b): a in A, b in B, a+b=k}| peaks near k = n+1 (the center of the sum range). A greedy approach that tracks the running overlap profile can make locally optimal decisions to spread the load across sums, avoiding concentration at any single k.

### Strategy

1. **Greedy construction**: Process elements in a chosen order. For each element e, compute the incremental max-overlap if e goes to A vs B. Choose the assignment that keeps max overlap lowest. Break ties by balancing set sizes.

2. **Element orderings**: Try multiple orderings (center-out, random, edges-first) to find which gives the best greedy starting point.

3. **Local search refinement**: After greedy construction, iteratively try swapping an element from A with one from B. Accept swaps that reduce the max overlap. Continue until no improving swap exists.

4. **Multi-start**: Run multiple random orderings and pick the best result.

## Results

(Will be populated after evaluation runs)
