import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

file1 = '/project/atlas/users/sabdadin/output/training_samples/alices_alices_gaussian_prior_0_1.2_10000_thetas_10e7_samples_CP_odd/theta0_train_ratio_ggf_combined_signal_0.npy'
# Load data
data1 = np.load(file1)

bins = 2500  # Number of bins for the histogram
plt.figure(figsize=(10, 6))
plt.hist(data1, bins=bins, density=True, alpha=0.5, label="Theta 0", color="blue")
# Add labels and legend
plt.xlabel("Theta values")
plt.ylabel("Density")
plt.ylim(0,1)
plt.xlim(-4,4)
plt.title("Histograms of theta0_train_ratio")
plt.legend()
plt.tight_layout()
# Save and show the plot
plt.savefig("histograms_theta0_alices.png")