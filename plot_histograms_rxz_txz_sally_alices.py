import numpy as np
import matplotlib.pyplot as plt

# Load data for alices
# file_rxz_alices_SO = '/project/atlas/users/sabdadin/output/training_samples/alices_alices_gaussian_prior_0_1.2_10000_thetas_10e7_samples_CP_odd/r_xz_train_ratio_ggf_combined_signal_0.npy'
# data_rxz_alices_SO = np.load(file_rxz_alices_SO)
# llr_alices_SO = np.log(data_rxz_alices_SO)
# file_rxz_alices_SB = '/project/atlas/users/sabdadin/output/training_samples/alices_alices_gaussian_prior_0_1.2_10000_thetas_10e7_samples_CP_odd/r_xz_train_ratio_ggf_signal_plus_ttbar_0.npy'
# data_rxz_alices_SB = np.load(file_rxz_alices_SB)
# llr_alices_SB = np.log(data_rxz_alices_SB)

# bins_SO = np.linspace(-3,3,100)  # Number of bins for the histogram
# bins_SB = np.linspace(-3,3,100)

# plt.figure(figsize=(10, 6))

# # Signal Only 
# plt.hist(llr_alices_SO, bins=bins_SO, density=True, alpha=0.2, label="Signal Only", color="teal")
# # Signal + Background 
# plt.hist(llr_alices_SB, bins=bins_SB, density=True, alpha=0.2, label="Signal + Backgrounds", color="mediumblue")

# # Add labels and legend
# plt.xlabel(r'$ \log r(x,z|\theta_0,\theta_1)$', size=14)
# plt.ylabel("Density")
# plt.ylim(0,7)
# plt.xlim(-1.5, 1.5)
# plt.legend()
# plt.tight_layout()
# plt.savefig("histograms_logrxz_alices.png", dpi=600, bbox_inches='tight')

##########sally#############
file_rxz_sally_SB = '/project/atlas/users/sabdadin/output/training_samples/sally/t_xz_train_score_ggf_signal_plus_ttbar_0.npy'
data_rxz_sally_SB = np.load(file_rxz_sally_SB)

file_rxz_sally_SO = '/project/atlas/users/sabdadin/output/training_samples/sally/t_xz_train_score_ggf_combined_signal_0.npy'
data_rxz_sally_SO = np.load(file_rxz_sally_SO)

bins_SO = np.linspace(-0.15, 0.15,50)  # Number of bins for the histogram
bins_SB = np.linspace(-0.15, 0.15,50)

plt.figure(figsize=(10, 6))

# Signal Only 
plt.hist(data_rxz_sally_SO, bins=bins_SO, histtype='step', color='teal', density=True,linewidth=2)
plt.hist(data_rxz_sally_SO, bins=bins_SO, color='teal', density=True, alpha=0.2)
plt.plot(-0.5,0.5,label = "Signal Only", linewidth = 2, color = "teal")
# Signal + Background 
plt.hist(data_rxz_sally_SB, bins=bins_SB, histtype='step', color='mediumblue', density=True,linewidth=2)
plt.hist(data_rxz_sally_SB, bins=bins_SB, color='mediumblue', density=True, alpha=0.2)
plt.plot(-0.5,0.5,label = "Signal + Backgrounds", linewidth = 2, color = "mediumblue")

# Add labels and legend
plt.xlabel(r'$ t(x,z|\theta_0,\theta_1)$', size=14)
plt.ylabel("Density",size=14)
# plt.ylim(0,50)
plt.xlim(-0.1, 0.1)
plt.legend(frameon=False, fontsize=11)
plt.tight_layout()
plt.savefig("histograms_txz_sally.png", dpi=600, bbox_inches='tight')
