"""Generate teaser figure for the Erdos Minimum Overlap campaign."""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "medium",
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.constrained_layout.use": True,
})

COLORS = {
    "random": "#C44E52",
    "lattice": "#8172B3",
    "slp": "#4C72B0",
    "slp_lattice": "#55A868",
    "best": "#DD8452",
    "lower_bound": "#888888",
}

n = 500
rng = np.random.RandomState(42)

# Random partition overlap profile
perm = rng.permutation(2 * n) + 1
A_rand = set(perm[:n].tolist())
B_rand = set(perm[n:].tolist())

f_A = np.zeros(2 * n + 1)
f_B = np.zeros(2 * n + 1)
for a in A_rand:
    f_A[a] = 1.0
for b in B_rand:
    f_B[b] = 1.0
conv_rand = np.convolve(f_A, f_B) / n

# Interleaved partition (A=odds, B=evens) - a structured approach
A_inter = set(range(1, 2 * n + 1, 2))
B_inter = set(range(2, 2 * n + 1, 2))
f_A2 = np.zeros(2 * n + 1)
f_B2 = np.zeros(2 * n + 1)
for a in A_inter:
    f_A2[a] = 1.0
for b in B_inter:
    f_B2[b] = 1.0
conv_inter = np.convolve(f_A2, f_B2) / n

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel (a): overlap profiles
ax = axes[0]
ks = np.arange(len(conv_rand))
ax.plot(ks, conv_rand, color=COLORS["random"], alpha=0.7, linewidth=0.8, label=f"Random (max={conv_rand.max():.3f})")
ax.plot(np.arange(len(conv_inter)), conv_inter, color=COLORS["slp"], alpha=0.7, linewidth=0.8, label=f"Interleaved (max={conv_inter.max():.3f})")
ax.axhline(y=0.380926, color=COLORS["lower_bound"], linestyle="--", linewidth=1.2, label="Lower bound (0.381)")
ax.set_xlabel("Sum k")
ax.set_ylabel("Normalized overlap r(k)/n")
ax.set_title("(a) Overlap profiles for two partitions")
ax.legend(loc="upper right", fontsize=9)
ax.set_xlim(0, 2 * (2 * n))

# Panel (b): known results landscape
ax2 = axes[1]
methods = ["Random", "Lattice\npack", "Cilleruelo\n(2018)", "SLP-256", "SLP-1024", "SLP-lit", "SLP-\nlattice", "SLP-lattice\n-anneal"]
metrics = [0.492, 0.403, 0.411, 0.452, 0.391, 0.389, 0.382, 0.38087]
colors_bar = [COLORS["random"], COLORS["lattice"], COLORS["lattice"],
              COLORS["slp"], COLORS["slp"], COLORS["slp"],
              COLORS["slp_lattice"], COLORS["best"]]

bars = ax2.bar(range(len(methods)), metrics, color=colors_bar, alpha=0.85, width=0.7)
ax2.axhline(y=0.380926, color=COLORS["lower_bound"], linestyle="--", linewidth=1.2, label="Lower bound (0.381)")
ax2.set_xticks(range(len(methods)))
ax2.set_xticklabels(methods, fontsize=8.5)
ax2.set_ylabel("Normalized max overlap")
ax2.set_title("(b) Known results landscape")
ax2.set_ylim(0.37, 0.52)
ax2.legend(loc="upper right", fontsize=9)

# Annotate best
ax2.annotate(f"{metrics[-1]:.5f}", xy=(len(methods)-1, metrics[-1]),
             xytext=(len(methods)-2.5, metrics[-1]+0.02),
             fontsize=9, arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

fig.suptitle("Erdos Minimum Overlap Problem", fontsize=14, fontweight="medium", y=1.02)
plt.savefig("research/figures/teaser.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Teaser figure saved to research/figures/teaser.png")
