"""Generate figures for orbit/02-fourier-optimize."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import importlib.util

# Style
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
    'sa_max': '#4C72B0',
    'sa_topk': '#DD8452',
    'theoretical': '#C44E52',
}

n = 1000
size = 2 * n
r_size = 2 * size + 2

def build_overlap(A_set):
    A = np.array(sorted(A_set), dtype=np.int32)
    B = np.array(sorted(set(range(1, size+1)) - A_set), dtype=np.int32)
    f_A = np.zeros(size+1)
    f_B = np.zeros(size+1)
    f_A[A] = 1.0
    f_B[B] = 1.0
    conv = np.convolve(f_A, f_B)
    return conv

# Load solution
spec = importlib.util.spec_from_file_location("solution",
    os.path.join(os.path.dirname(__file__), "solution.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Generate partitions
rng = np.random.default_rng(42)

# Random partition
perm_rand = rng.permutation(size) + 1
A_random = set(perm_rand[:n].tolist())
r_random = build_overlap(A_random)

# Our solution
A_topk = mod.solve(n=n, seed=42)
r_topk = build_overlap(A_topk)

# SA with max-only (simulate by using a fixed random partition refined partially)
# Just use a different random start to show what random looks like
rng2 = np.random.default_rng(123)
perm2 = rng2.permutation(size) + 1
A_random2 = set(perm2[:n].tolist())
r_random2 = build_overlap(A_random2)

metric_random = float(np.max(r_random)) / n
metric_topk = float(np.max(r_topk)) / n

# === FIGURE: Multi-panel ===
fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

# Panel (a): Overlap profiles comparison
ax = axes[0, 0]
ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold')

sums = np.arange(len(r_random))
ax.plot(sums, r_random, color=COLORS['random'], alpha=0.5, linewidth=0.5,
        label=f'Random (metric={metric_random:.3f})')
ax.plot(sums, r_topk, color=COLORS['sa_topk'], alpha=0.7, linewidth=0.5,
        label=f'SA+TopK (metric={metric_topk:.3f})')
ax.axhline(y=0.381 * n, color=COLORS['theoretical'], linestyle='--', linewidth=1.5,
           label='Theoretical lower bound (0.381n)')
ax.set_xlabel('Sum k')
ax.set_ylabel('Overlap count r(k)')
ax.set_title('Overlap profiles: random vs optimized partition')
ax.set_xlim(0, 2*size)
ax.legend(loc='upper right')

# Panel (b): Zoom into peak region
ax = axes[0, 1]
ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold')

# Focus on the peak region (around n+1)
peak_center = n + 1
window = 200
lo, hi = peak_center - window, peak_center + window

ax.plot(sums[lo:hi], r_random[lo:hi], color=COLORS['random'], alpha=0.5,
        linewidth=1, label='Random')
ax.plot(sums[lo:hi], r_topk[lo:hi], color=COLORS['sa_topk'], alpha=0.7,
        linewidth=1, label='SA+TopK')
ax.axhline(y=0.381 * n, color=COLORS['theoretical'], linestyle='--', linewidth=1.5,
           label='Lower bound')
ax.set_xlabel('Sum k')
ax.set_ylabel('Overlap count r(k)')
ax.set_title('Peak region detail (k near n+1)')
ax.legend(loc='upper right')

# Panel (c): Partition visualization (which elements are in A)
ax = axes[1, 0]
ax.text(-0.12, 1.05, '(c)', transform=ax.transAxes, fontsize=14, fontweight='bold')

A_arr = np.array(sorted(A_topk))
indicator = np.zeros(size + 1)
indicator[A_arr] = 1.0

# Show as a heatmap-style strip
ax.fill_between(range(1, size+1), 0, indicator[1:],
                color=COLORS['sa_topk'], alpha=0.3, linewidth=0)
ax.set_xlabel('Element i')
ax.set_ylabel('In set A')
ax.set_title('Optimized partition: elements assigned to A')
ax.set_xlim(1, size)
ax.set_ylim(-0.1, 1.3)
ax.set_yticks([0, 1])
ax.set_yticklabels(['B', 'A'])

# Panel (d): Histogram of overlap values
ax = axes[1, 1]
ax.text(-0.12, 1.05, '(d)', transform=ax.transAxes, fontsize=14, fontweight='bold')

# Only count non-zero overlaps
r_random_nz = r_random[r_random > 0]
r_topk_nz = r_topk[r_topk > 0]

bins = np.arange(0, max(np.max(r_random_nz), np.max(r_topk_nz)) + 10, 5)
ax.hist(r_random_nz, bins=bins, alpha=0.5, color=COLORS['random'],
        label='Random', density=True)
ax.hist(r_topk_nz, bins=bins, alpha=0.5, color=COLORS['sa_topk'],
        label='SA+TopK', density=True)
ax.axvline(x=0.381 * n, color=COLORS['theoretical'], linestyle='--', linewidth=1.5,
           label='Lower bound')
ax.set_xlabel('Overlap count r(k)')
ax.set_ylabel('Density')
ax.set_title('Distribution of overlap values')
ax.legend(loc='upper right')

fig.suptitle('Erdos Minimum Overlap: Fourier-Guided SA Optimization', y=1.02,
             fontsize=14, fontweight='medium')

fig_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(fig_dir, exist_ok=True)
fig.savefig(os.path.join(fig_dir, "results.png"), dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)

print(f"Random metric: {metric_random:.4f}")
print(f"SA+TopK metric: {metric_topk:.4f}")
print("Figure saved to figures/results.png")
