#!/usr/bin/env python3
"""Generate figures for orbit/01-interleave-refine."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import solution

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "medium",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 10.0,
    "axes.labelpad": 6.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.borderpad": 0.3,
    "legend.handletextpad": 0.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.constrained_layout.use": True,
})

COLORS = {
    'random': '#888888',
    'sa_hardmax': '#DD8452',
    'sa_l2': '#4C72B0',
    'theoretical': '#C44E52',
}

n = 1000
N = 2 * n

def compute_profile(A_set, n):
    N = 2 * n
    f_A = np.zeros(N + 1)
    f_B = np.zeros(N + 1)
    for a in A_set:
        f_A[a] = 1.0
    for x in range(1, N + 1):
        if f_A[x] == 0:
            f_B[x] = 1.0
    conv = np.convolve(f_A, f_B)
    return conv

# Generate solutions
print("Generating random partition...")
rng = np.random.RandomState(42)
perm = rng.permutation(np.arange(1, N + 1))
A_random = set(perm[:n].tolist())
profile_random = compute_profile(A_random, n)

print("Generating L2-annealed partition (this takes ~42s)...")
A_l2 = solution.solve(n, seed=1)
profile_l2 = compute_profile(A_l2, n)

print("Computing metrics...")
metric_random = np.max(profile_random) / n
metric_l2 = np.max(profile_l2) / n

print(f"Random: {metric_random:.4f}")
print(f"L2-SA: {metric_l2:.4f}")

# --- Figure: 3-panel ---
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Panel (a): Overlap profiles comparison
ax = axes[0]
ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold')
k_range = np.arange(len(profile_random))
# Subsample for plotting
mask = (k_range >= 2) & (k_range <= 4 * n)
ax.fill_between(k_range[mask], profile_random[mask] / n, alpha=0.15, color=COLORS['random'])
ax.plot(k_range[mask], profile_random[mask] / n, color=COLORS['random'],
        linewidth=0.5, alpha=0.7, label=f'Random ({metric_random:.3f})')
ax.fill_between(k_range[mask], profile_l2[mask] / n, alpha=0.15, color=COLORS['sa_l2'])
ax.plot(k_range[mask], profile_l2[mask] / n, color=COLORS['sa_l2'],
        linewidth=0.5, alpha=0.7, label=f'L2-SA ({metric_l2:.3f})')
ax.axhline(y=0.3809, color=COLORS['theoretical'], linestyle='--', linewidth=1.0,
           label='Lower bound (0.381)')
ax.set_xlabel('Sum k')
ax.set_ylabel('Normalized overlap r(k)/n')
ax.set_title('Overlap profiles')
ax.legend(loc='upper right')
ax.set_ylim(0, 0.6)

# Panel (b): Partition visualization (membership indicator)
ax = axes[1]
ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold')

# Show which elements are in A for L2-SA partition
elements = np.arange(1, N + 1)
in_A = np.array([1 if x in A_l2 else 0 for x in elements])

# Reshape into a grid for visualization
side = 50  # 50x40 = 2000
cols = 40
img = in_A.reshape(side, cols)
ax.imshow(img, cmap='RdBu', aspect='auto', interpolation='nearest',
          extent=[1, cols, side, 1])
ax.set_xlabel('Column (element mod 40)')
ax.set_ylabel('Row (element / 40)')
ax.set_title('L2-SA partition (blue=A, red=B)')
ax.grid(False)

# Panel (c): Top overlap values comparison
ax = axes[2]
ax.text(-0.12, 1.05, '(c)', transform=ax.transAxes, fontsize=14, fontweight='bold')

# Sort overlap values descending and plot top 50
top_n = 100
sorted_random = np.sort(profile_random[mask])[::-1][:top_n] / n
sorted_l2 = np.sort(profile_l2[mask])[::-1][:top_n] / n

ax.plot(range(top_n), sorted_random, color=COLORS['random'], linewidth=1.5,
        label=f'Random (max={metric_random:.3f})')
ax.plot(range(top_n), sorted_l2, color=COLORS['sa_l2'], linewidth=1.5,
        label=f'L2-SA (max={metric_l2:.3f})')
ax.axhline(y=0.3809, color=COLORS['theoretical'], linestyle='--', linewidth=1.0,
           label='Lower bound')
ax.set_xlabel('Rank (sorted descending)')
ax.set_ylabel('Normalized overlap r(k)/n')
ax.set_title('Top overlap values (sorted)')
ax.legend(loc='upper right')
ax.set_ylim(0.3, 0.6)

fig.suptitle('Erdos Minimum Overlap: L2-surrogate SA vs Random Partition (n=1000)',
             fontsize=14, fontweight='medium', y=1.02)

figpath = os.path.join(os.path.dirname(__file__), 'figures', 'results.png')
fig.savefig(figpath, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved {figpath}")
