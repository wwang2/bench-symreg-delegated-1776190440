#!/bin/bash
# Reproduce orbit/02-fourier-optimize results
# Run from worktree root: .worktrees/02-fourier-optimize/

echo "=== Evaluating solution ==="
for SEED in 1 2 3; do
    echo -n "Seed $SEED: "
    python3 research/eval/evaluator.py --solution orbits/02-fourier-optimize/solution.py --seed $SEED
done

echo ""
echo "=== Generating figures ==="
python3 orbits/02-fourier-optimize/make_figures.py
