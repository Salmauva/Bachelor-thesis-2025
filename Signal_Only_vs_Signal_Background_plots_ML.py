import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import UnivariateSpline
import matplotlib.colors as colors
import matplotlib as mpl
def calculate_std(load_dir):

    predictions = []

    for i in range(5):
    
      estimator_number = i + 1
      dir =  load_dir + f"/estimator_{i}_data.npz"

      data = np.load(dir)

      parameter_grid = data['parameter_grid']
      llr_kin = data['llr_kin']
      llr_rate = data['llr_rate']
      index_best_point = data['index_best_point']

      rescaled_log_r = llr_kin+llr_rate
      rescaled_log_r = -2.0*(rescaled_log_r[:] - rescaled_log_r[index_best_point])    
      predictions.append(rescaled_log_r)

    std_deviation = np.std(predictions, axis=0)

    return std_deviation
plot_dir = '/project/atlas/users/sabdadin/output/plots'
fig_name = "signal_vs_background_alice_method"

# Load data (assuming functions for loading the std are already defined)
load_dir_SB = "/project/atlas/users/sabdadin/output/llr_fits/alices_gaussian_prior_0_1.2_10000_thetas_10e7_samples_CP_odd/kinematic_only/alices_hidden_[100, 100]_tanh_alpha_0_epochs_100_bs_128/ggf_signal_plus_ttbar/range_[-0.2, 0.2]_resolutions_801"
load_dir_S = "/project/atlas/users/sabdadin/output/llr_fits/alices_gaussian_prior_0_1.2_10000_thetas_10e7_samples_CP_odd/kinematic_only/alices_hidden_[50]_tanh_alpha_0_epochs_50_bs_128/ggf_combined_signal/range_[-0.2, 0.2]_resolutions_801"
data_SB = np.load(f"{load_dir_SB}/ensemble_data.npz")
data_S = np.load(f"{load_dir_S}/ensemble_data.npz")

parameter_grid_SB = np.squeeze(data_SB['parameter_grid'])
llr_kin_SB = data_SB['llr_kin']
llr_rate_SB = data_SB['llr_rate']
index_best_point_SB = data_SB['index_best_point']

parameter_grid_S = np.squeeze(data_S['parameter_grid'])
llr_kin_S = data_S['llr_kin']
llr_rate_S = data_S['llr_rate']
index_best_point_S = data_S['index_best_point']
# Calculate rescaled log-likelihood ratios
rescaled_log_r_SB = llr_kin_SB + llr_rate_SB
rescaled_log_r_SB = -2.0 * (rescaled_log_r_SB[:] - rescaled_log_r_SB[index_best_point_SB])
rescaled_log_r_S = llr_kin_S + llr_rate_S
rescaled_log_r_S = -2.0 * (rescaled_log_r_S[:] - rescaled_log_r_S[index_best_point_S])

# Load standard deviations (assuming calculate_std is defined)
std_SB = np.squeeze(calculate_std(load_dir_SB))
std_S = np.squeeze(calculate_std(load_dir_S))
# Spline interpolation for signal only
spline_S = UnivariateSpline(parameter_grid_S, rescaled_log_r_S, s=0)
spline_S_std_low= UnivariateSpline(parameter_grid_S, rescaled_log_r_S - std_S, s=0)
spline_S_std_high = UnivariateSpline(parameter_grid_S, rescaled_log_r_S + std_S, s=0)

# Create a finer grid with 1001 points for smooth interpolation
num_points = 10000
x_interp = np.linspace(-1.2, 1.2, num_points)
#spline interpolation for signal + backgrounds
spline_SB = UnivariateSpline(parameter_grid_SB, rescaled_log_r_SB, s=0)
spline_SB_std_low= UnivariateSpline(parameter_grid_SB, rescaled_log_r_SB - std_SB, s=0)
spline_SB_std_high = UnivariateSpline(parameter_grid_SB, rescaled_log_r_SB + std_SB, s=0)

# Create a finer grid with 1001 points for smooth interpolation
num_points = 10000
x_interp = np.linspace(-1.2, 1.2, num_points)


# Plotting
plt.figure(figsize=(7, 6))
plt.axhline(y=1.0, color='dimgray', linestyle='-.', linewidth=1.2)
plt.axhline(y=3.84, color='dimgray', linestyle='--', linewidth=1.2)

# Adding text labels directly on the lines
plt.text(-0.02, 1.0, '68% CL', color='dimgray', verticalalignment='bottom', horizontalalignment='left', fontsize=11)
plt.text(-0.02, 3.84, '95% CL', color='dimgray', verticalalignment='bottom', horizontalalignment='left', fontsize=11)



# Signal only
# plt.plot(parameter_grid_S, rescaled_log_r_S, lw=1.5, color="teal", label="Signal Only")
plt.plot(x_interp, spline_S(x_interp), lw=1.5, color="teal", label="Signal Only")
plt.fill_between(x_interp, spline_S_std_low(x_interp), spline_S_std_high(x_interp), color="teal", alpha=0.1)


# Signal + Backgrounds
# plt.plot(parameter_grid_SB, rescaled_log_r_SB , lw=1.5, color="indigo", label="Signal + Backgrounds")
plt.plot(x_interp, spline_SB(x_interp), lw=1.5, color="indigo", label="Signal + Backgrounds")
plt.fill_between(x_interp, spline_SB_std_low(x_interp), spline_SB_std_high(x_interp), color="indigo", alpha=0.1)

# Labels, legend, and limits
plt.xlabel(r"$c_{H\tilde{G}}$", size=14)
plt.ylabel(r"$q(\theta)$", size=14)
# plt.text(-0.003,8,r'$\bf{ALICE}$' + r'$\ (\alpha = 0, n_{\theta_0} = 1000)$',fontsize=12,bbox=dict(facecolor='white', edgecolor='none', alpha=0.5),horizontalalignment='center')
title = (r'$\bf{ALICE}$' + r'$\ (\alpha = 0, n_{\theta_0} = 1000)$')
luminosity_info = (
    r"$h \rightarrow W^+ + W^- \rightarrow \ell^+ \nu_\ell\, \ell^- \nu_\ell, \mathcal{L} = 300\, \mathrm{fb}^{-1}$"
)
text = title + "\n" + luminosity_info
plt.text(-0.003,8,text,fontsize=12,bbox=dict(facecolor='white', edgecolor='none', alpha=0.5),horizontalalignment='center')
plt.legend( frameon=False, loc="upper center", fontsize=12)

plt.ylim(0, 10)
plt.xlim(-0.02, 0.02)  # Adjusted limits to match the new range
plt.savefig(f"{plot_dir}/{fig_name}.png",dpi=600,bbox_inches='tight' )