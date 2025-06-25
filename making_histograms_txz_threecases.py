import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# File paths
file1 = "/project/atlas/users/sabdadin/output/training_samples/sally/t_xz_train_score_ggf_combined_signal_0.npy"
file2 = "/project/atlas/users/sabdadin/output/training_samples/sally/t_xz_train_score_ggf_signal_plus_ttbar_0.npy"
file3 = "/project/atlas/users/sabdadin/output/training_samples/sally/t_xz_train_score_ttbar_0.npy"

# Load data
data1 = np.load(file1)
data2 = np.load(file2)
data3 = np.load(file3)

# Plot histograms
bins = 2500  # Number of bins for the histogram
plt.figure(figsize=(10, 6))
plt.hist(data1, bins=bins, density=True, alpha=0.5, label="ggf_combined_signal", color="blue")
plt.hist(data2, bins=bins, density=True, alpha=0.5, label="ggf_signal_plus_ttbar", color="orange")
plt.hist(data3, bins=bins, density=True, alpha=0.5, label="ggf_ttbar", color="green")

# Add labels and legend
plt.xlabel("Score")
plt.ylabel("Density")
plt.ylim(0,5)
plt.xlim(-5,5)
plt.title("Histograms of t_xz_train_score")
plt.legend()
plt.tight_layout()

# Save and show the plot
plt.savefig("histograms_txz_threecases.png")
