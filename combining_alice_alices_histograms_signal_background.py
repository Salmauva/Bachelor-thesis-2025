import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
def calculate_std(file_path):
    """
    Calculates the standard deviation for the given `.npz` file.
    """
    predictions = []

    for i in range(5):
        # Construct the file path for each estimator
        estimator_file = f"{file_path}/estimator_{i}_data.npz"

        # Load the data
        data = np.load(estimator_file)

        parameter_grid = data['parameter_grid']
        llr_kin = data['llr_kin']
        llr_rate = data['llr_rate']
        index_best_point = data['index_best_point']

        # Calculate rescaled log-likelihood ratio
        rescaled_log_r = llr_kin + llr_rate
        rescaled_log_r = -2.0 * (rescaled_log_r[:] - rescaled_log_r[index_best_point])
        predictions.append(rescaled_log_r)

    # Calculate the standard deviation across predictions
    std_deviation = np.std(predictions, axis=0)

    return std_deviation
#######alice######
load_dir_SB_alice = "/project/atlas/users/sabdadin/output/llr_fits/alices_gaussian_prior_0_1.2_10000_thetas_10e7_samples_CP_odd/kinematic_only/alices_hidden_[100, 100]_tanh_alpha_0_epochs_50_bs_128/ggf_signal_plus_ttbar/range_[-0.2, 0.2]_resolutions_801"

data_SB_alice = np.load(f"{load_dir_SB_alice}/ensemble_data.npz")

parameter_grid_SB_alice = np.squeeze(data_SB_alice['parameter_grid'])
llr_kin_SB_alice = data_SB_alice['llr_kin']
llr_rate_SB_alice = data_SB_alice['llr_rate']
index_best_point_SB_alice = data_SB_alice['index_best_point']

# Calculate rescaled log-likelihood ratios
rescaled_log_r_SB_alice = llr_kin_SB_alice + llr_rate_SB_alice
rescaled_log_r_SB_alice = -2.0 * (rescaled_log_r_SB_alice[:] - rescaled_log_r_SB_alice[index_best_point_SB_alice])

# Load standard deviations (assuming calculate_std is defined)
std_SB_alice = np.squeeze(calculate_std(load_dir_SB_alice))

# Spline interpolation for signal only
spline_SB_alice = UnivariateSpline(parameter_grid_SB_alice, rescaled_log_r_SB_alice, s=0)
spline_SB_std_low_alice = UnivariateSpline(parameter_grid_SB_alice, rescaled_log_r_SB_alice - std_SB_alice, s=0)
spline_SB_std_high_alice = UnivariateSpline(parameter_grid_SB_alice, rescaled_log_r_SB_alice + std_SB_alice, s=0)

# Create a finer grid with 1001 points for smooth interpolation
num_points_alice = 10000
x_interp_alice = np.linspace(-1.2, 1.2, num_points_alice)

####alices,alpha1######

load_dir_S_alices = "/project/atlas/users/sabdadin/output/llr_fits/alices_gaussian_prior_0_1.2_10000_thetas_10e7_samples_CP_odd/kinematic_only/alices_hidden_[100, 100]_tanh_alpha_1_epochs_100_bs_128/ggf_signal_plus_ttbar/range_[-0.2, 0.2]_resolutions_801"

data_S_alices = np.load(f"{load_dir_S_alices}/ensemble_data.npz")

parameter_grid_S_alices = np.squeeze(data_S_alices['parameter_grid'])
llr_kin_S_alices = data_S_alices['llr_kin']
llr_rate_S_alices = data_S_alices['llr_rate']
index_best_point_S_alices = data_S_alices['index_best_point']

# Calculate rescaled log-likelihood ratios
rescaled_log_r_S_alices = llr_kin_S_alices + llr_rate_S_alices
rescaled_log_r_S_alices = -2.0 * (rescaled_log_r_S_alices[:] - rescaled_log_r_S_alices[index_best_point_S_alices])

# Load standard deviations (assuming calculate_std is defined)
std_S_alices = np.squeeze(calculate_std(load_dir_S_alices))

# Spline interpolation for signal only
spline_S_alices = UnivariateSpline(parameter_grid_S_alices, rescaled_log_r_S_alices, s=0)
spline_S_std_low_alices = UnivariateSpline(parameter_grid_S_alices, rescaled_log_r_S_alices - std_S_alices, s=0)
spline_S_std_high_alices = UnivariateSpline(parameter_grid_S_alices, rescaled_log_r_S_alices + std_S_alices, s=0)

# Create a finer grid with 1001 points for smooth interpolation
num_points_alices = 10000
x_interp_alices = np.linspace(-1.2, 1.2, num_points_alices)

################################hist2D
load_dir_SB_hist = "/project/atlas/users/sabdadin/output/llr_fits_hist/dphi_jj/ggf_signal_plus_ttbar/range_[[-2, 2]]_resolutions_[5000]/data_1D_Ricardos_binning_2D_case.npz"
load_dir_S_hist= "/project/atlas/users/sabdadin/output/llr_fits_hist/dphi_jj/ggf_combined_signal/range_[[-2, 2]]_resolutions_[5000]/data_1D_Ricardos_binning_2D_case.npz"

data_SB_hist = np.load(f"{load_dir_SB_hist}")
data_S_hist = np.load(f"{load_dir_S_hist}")

parameter_grid_SB_hist = np.squeeze(data_SB_hist['parameter_grid'])
llr_kin_SB_hist = data_SB_hist['llr_kin']
llr_rate_SB_hist = data_SB_hist['llr_rate']
index_best_point_SB_hist = data_SB_hist['index_best_point']

parameter_grid_S_hist = np.squeeze(data_S_hist['parameter_grid'])
llr_kin_S_hist = data_S_hist['llr_kin']
llr_rate_S_hist = data_S_hist['llr_rate']
index_best_point_S_hist = data_S_hist['index_best_point']

# Calculate rescaled log-likelihood ratios
rescaled_log_r_SB_hist = llr_kin_SB_hist + llr_rate_SB_hist
rescaled_log_r_SB_hist = -2.0 * (rescaled_log_r_SB_hist[:] - rescaled_log_r_SB_hist[index_best_point_SB_hist])
rescaled_log_r_S_hist = llr_kin_S_hist + llr_rate_S_hist
rescaled_log_r_S_hist = -2.0 * (rescaled_log_r_S_hist[:] - rescaled_log_r_S_hist[index_best_point_S_hist])


# Spline interpolation for signal only
spline_S_hist = UnivariateSpline(parameter_grid_S_hist, rescaled_log_r_S_hist, s=0)

# Spline interpolation for signal + backgrounds
spline_SB_hist = UnivariateSpline(parameter_grid_SB_hist, rescaled_log_r_SB_hist, s=0)

# Create a finer grid with 1001 points for smooth interpolation
num_points_hist = 10000
x_interp_hist = np.linspace(-1.2, 1.2, num_points_hist)
################################hist1D
load_dir_SB_hist_1d = "/project/atlas/users/sabdadin/output/llr_fits_hist/dphi_jj/ggf_signal_plus_ttbar/range_[[-2, 2]]_resolutions_[5000]/data_1D_Ricardos_binning_2D_case.npz"
load_dir_S_hist_1d= "/project/atlas/users/sabdadin/output/llr_fits_hist/dphi_jj/ggf_combined_signal/range_[[-2, 2]]_resolutions_[5000]/data_1D_Ricardos_binning_2D_case.npz"

data_SB_hist_1d = np.load(f"{load_dir_SB_hist_1d}")
data_S_hist_1d = np.load(f"{load_dir_S_hist_1d}")

parameter_grid_SB_hist_1d = np.squeeze(data_SB_hist_1d['parameter_grid'])
llr_kin_SB_hist_1d = data_SB_hist_1d['llr_kin']
llr_rate_SB_hist_1d = data_SB_hist_1d['llr_rate']
index_best_point_SB_hist_1d = data_SB_hist_1d['index_best_point']

parameter_grid_S_hist_1d = np.squeeze(data_S_hist_1d['parameter_grid'])
llr_kin_S_hist_1d = data_S_hist['llr_kin']
llr_rate_S_hist_1d = data_S_hist['llr_rate']
index_best_point_S_hist_1d = data_S_hist_1d['index_best_point']

# Calculate rescaled log-likelihood ratios
rescaled_log_r_SB_hist_1d = llr_kin_SB_hist_1d + llr_rate_SB_hist_1d
rescaled_log_r_SB_hist_1d = -2.0 * (rescaled_log_r_SB_hist_1d[:] - rescaled_log_r_SB_hist_1d[index_best_point_SB_hist_1d])
rescaled_log_r_S_hist_1d = llr_kin_S_hist_1d + llr_rate_S_hist_1d
rescaled_log_r_S_hist_1d = -2.0 * (rescaled_log_r_S_hist_1d[:] - rescaled_log_r_S_hist_1d[index_best_point_S_hist_1d])


# Spline interpolation for signal only
spline_S_hist_1d = UnivariateSpline(parameter_grid_S_hist_1d, rescaled_log_r_S_hist_1d, s=0)

# Spline interpolation for signal + backgrounds
spline_SB_hist_1d = UnivariateSpline(parameter_grid_SB_hist_1d, rescaled_log_r_SB_hist_1d, s=0)

# Create a finer grid with 1001 points for smooth interpolation
num_points_hist_1d = 10000
x_interp_hist_1d = np.linspace(-1.2, 1.2, num_points_hist_1d)

#####plotting#####
# Plotting
plt.figure(figsize=(7, 6))

# Signal only
# # Add horizontal lines and labels for CL
plt.axhline(y=1.0, color='dimgray', linestyle='-.', linewidth=1.2)
plt.axhline(y=3.84, color='dimgray', linestyle='--', linewidth=1.2)

# Adding text labels directly on the lines
plt.text(-0.02, 1.0, '68% CL', color='dimgray', verticalalignment='bottom', horizontalalignment='left', fontsize=11)
plt.text(-0.02, 3.84, '95% CL', color='dimgray', verticalalignment='bottom', horizontalalignment='left', fontsize=11)

plt.plot(x_interp_alices, spline_S_alices(x_interp_alices), lw=1.5, color="mediumblue", label="ALICES")
plt.fill_between(x_interp_alices, spline_S_std_low_alices(x_interp_alices), spline_S_std_high_alices(x_interp_alices), color="mediumblue", alpha=0.1)

plt.plot(x_interp_alice, spline_SB_alice(x_interp_alice), lw=1.5, color="darkgreen", label="ALICE")
plt.fill_between(x_interp_alice, spline_SB_std_low_alice(x_interp_alice), spline_SB_std_high_alice(x_interp_alice), color="darkgreen", alpha=0.1)

plt.plot(x_interp_hist, spline_SB_hist(x_interp_hist), lw=1.5, color="mediumvioletred", label=r'$\Delta \Phi_{jj} \otimes \Delta  \Phi_{ll}$')
plt.plot(x_interp_hist_1d, spline_SB_hist_1d(x_interp_hist_1d), lw=1.5, ls='--',color="#ee9d33", label=r'$\Delta \Phi_{jj}$')
plt.xlim(-0.02,0.02)
plt.ylim(0,10)
plt.xlabel(r"$c_{H \tilde G}$", size=14)
plt.ylabel(r"$q(\theta)$", size=14)
luminosity_info = (
    r"$h \rightarrow W^+ + W^- \rightarrow \ell^+ \nu_\ell\, \ell^- \nu_\ell$,  SB"
    "\n"
    r"$\mathcal{L} = 300\, \mathrm{fb}^{-1}$"
)

plt.text(-0.0003, 8, luminosity_info,
         fontsize=12,   # Center the text horizontally
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.5),horizontalalignment='center')
plt.legend(fontsize=12,frameon=False)

plt.savefig("/project/atlas/users/sabdadin/output/plots/CP_odd_signal_background_all_methods.png",dpi=600,bbox_inches='tight')
