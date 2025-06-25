import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import UnivariateSpline
import matplotlib.colors as colors
import matplotlib as mpl
def calculate_std(load_dir):

    data = np.load(load_dir+'.npz')
    parameter_grid = data['parameter_grid']
    llr_kin = data['llr_kin']
    llr_rate = data['llr_rate']
    index_best_point = data['index_best_point']

    rescaled_log_r = llr_kin + llr_rate
    rescaled_log_r = -2.0 * (rescaled_log_r[:] - rescaled_log_r[index_best_point])
    std_deviation = np.std(rescaled_log_r)

    return std_deviation
plot_dir = '/project/atlas/users/sabdadin/output/plots'
fig_name = "signal_vs_background_histogram_method_2D_case_dphijj_phill_std_included"

# Load data (assuming functions for loading the std are already defined)
load_dir_SB = "/project/atlas/users/sabdadin/output/llr_fits_hist/dphi_jj/ggf_signal_plus_ttbar/range_[[-2, 2]]_resolutions_[5000]/data_1D_Ricardos_binning_2D_case"
load_dir_S = "/project/atlas/users/sabdadin/output/llr_fits_hist/dphi_jj/ggf_combined_signal/range_[[-2, 2]]_resolutions_[5000]/data_1D_Ricardos_binning_2D_case"
data_SB = np.load(f"{load_dir_SB}.npz")
data_S = np.load(f"{load_dir_S}.npz")

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
# Plotting
plt.figure(figsize=(7, 6))
# Signal only
plt.plot(parameter_grid_S, rescaled_log_r_S, lw=1.5, color="teal", label="Signal Only")
plt.fill_between(parameter_grid_S, rescaled_log_r_S, std_S, color="teal", alpha=0.1)

# Signal + Backgrounds
plt.plot(parameter_grid_SB, rescaled_log_r_SB , lw=1.5, color="indigo", label="Signal + Backgrounds")
plt.fill_between(parameter_grid_SB, rescaled_log_r_SB , std_SB, color="indigo", alpha=0.1)

# Labels, legend, and limits
plt.xlabel(r"$c_{H\tilde{W}}$", size=14)
plt.ylabel(r"$q(\theta)$", size=14)
plt.legend( frameon=False, loc="upper center", fontsize=12)

plt.ylim(-1, 500)
plt.xlim(-1.0, 1.0)  # Adjusted limits to match the new range
plt.savefig(f"{plot_dir}/{fig_name}.pdf",dpi=600,bbox_inches='tight' )