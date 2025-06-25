import numpy as np
import matplotlib.pyplot as plt

# Parameter name
parameter = r"$C_{HG\tilde{}}$"

# Methods and values
methods = ["AliceS", "Alice", "Sally", "1D hist", "2D hist"]
means = [0.0, -0.001, 0.00, 0.000, -0.000]

# 68% CL intervals (low, high) per method
ci_68 = {
    "AliceS": [-0.006, 0.006],
    "Alice":  [-0.005, 0.005],
    "Sally":  [-0.004, 0.004],
    "1D hist": [-0.006, 0.005],
    "2D hist": [-0.006, 0.005]
}

# 95% CL intervals (low, high) per method — dummy values, replace if needed
ci_95 = {
    "AliceS": [-0.009, 0.008],
    "Alice":  [-0.009, 0.008],
    "Sally":  [-0.007, 0.006],
    "1D hist": [-0.008, 0.008],
    "2D hist": [-0.008, 0.008]
}

# Vertical offsets and colors
offsets = [0.25, 0.125, 0.0, -0.125, -0.25]
colors = ['#009988', '#004488', '#EE7733', '#33BBEE', '#B759F8']

# Create figure
fig, ax = plt.subplots(figsize=(5, 3.5))

for i, method in enumerate(methods):
    bf = means[i]
    low68, high68 = ci_68[method]
    low95, high95 = ci_95[method]
    offset = offsets[i]

    # Plot 95% CI as dashed lines
    eb_95 = ax.errorbar(bf, offset,
                        xerr=[[bf - low95], [high95 - bf]],
                        fmt='o', color='black', ecolor=colors[i], elinewidth=1)
    eb_95[-1][0].set_linestyle('--')

    # Plot 68% CI as solid thicker lines
    eb_68 = ax.errorbar(bf, offset,
                        xerr=[[bf - low68], [high68 - bf]],
                        fmt='o', color='black', ecolor=colors[i], elinewidth=2)

# Y-axis setup
ax.set_yticks([0])
ax.set_yticklabels([parameter], fontsize=14)

# Decorations
ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
ax.set_xlabel("Parameter value", size=14)
ax.set_xlim(-0.04, 0.04)
ax.set_ylim(-0.35, 0.35)

# Manual legend
legend_lines = [
    plt.Line2D([0], [0], color=colors[0], lw=2, label='AliceS'),
    plt.Line2D([0], [0], color=colors[1], lw=2, label='Alice'),
    plt.Line2D([0], [0], color=colors[2], lw=2, label='Sally'),
    plt.Line2D([0], [0], color=colors[3], lw=2, label='1D hist'),
    plt.Line2D([0], [0], color=colors[4], lw=2, label='2D hist'),
]
ax.legend(handles=legend_lines, loc='upper right', fontsize=10)

ax.set_title("Signal Only", fontsize=13)
plt.tight_layout()
plt.savefig("summary_1param_5methods.pdf", dpi=600, bbox_inches='tight')

