import numpy as np
import matplotlib.pyplot as plt

mean = 0
stds_list = np.linspace(0,1,10)
stds = [1, 1.2, 1.5, 2]
x = np.linspace(-3, 3, 1000)

plt.figure(figsize=(8,5))
for std in stds:
    plt.plot(x, 1/(std*np.sqrt(2*np.pi)) * np.exp(-(x - mean)**2 / (2*std**2)), label=f'std={std}')
plt.axvline(-2.972, color='gray', linestyle='--', label='Morphing bounds')
plt.axvline(2.868, color='gray', linestyle='--')
plt.title("Comparison of Gaussian Priors")
plt.xlabel("Parameter Value")
plt.ylabel("Probability Density")
plt.ylim (0, 2)
plt.legend()
plt.grid(True)
plt.savefig("gaussian_priors.png")
######
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Mean and standard deviations to test
mean = 0.0
stds = [1, 1.2, 1.5, 2]  # Standard deviations for the Gaussian distributions

# X range
x = np.linspace(-3, 3, 1000)

plt.figure(figsize=(10, 6))

for std in stds:
    # Compute PDF
    y = norm.pdf(x, loc=mean, scale=std)
    plt.plot(x, y, label=f'Gaussian (σ={std})')

    # 66% confidence interval (about ±1σ for Gaussian)
    ci_66 = norm.interval(0.66, loc=mean, scale=std)
    plt.fill_between(x, 0, y, where=(x >= ci_66[0]) & (x <= ci_66[1]), alpha=0.1)

    # 99% confidence interval (about ±2.58σ)
    ci_99 = norm.interval(0.99, loc=mean, scale=std)
    plt.fill_between(x, 0, y, where=(x >= ci_99[0]) & (x <= ci_99[1]), alpha=0.05)

# Morphing bounds
plt.axvline(-2.972, color='gray', linestyle='--', label='Morphing bounds')
plt.axvline(2.868, color='gray', linestyle='--')

#plt.title("Gaussian Priors with 66% and 99% Confidence Intervals")
plt.xlabel("θ (e.g., $c_{H\\tilde{G}}$)")
plt.ylabel("Probability Density")
plt.legend()
plt.ylim (0, 0.5)
plt.grid(True)
plt.tight_layout()
plt.savefig("gaussian_priors_with_ci.png")