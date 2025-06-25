import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import UnivariateSpline
import matplotlib.colors as colors
import matplotlib as mpl

def find_confidence_limits(x_interp, rescaled_log_r_interp, threshold):
    # Find the x-values where the interpolated LLR crosses the confidence level threshold
    crossings = np.where(np.diff(np.sign(rescaled_log_r_interp - threshold)))[0]
    if len(crossings) >= 2:  # Need at least two crossings (one on either side of the best-fit point)
        lower_limit = x_interp[crossings[0]]
        upper_limit = x_interp[crossings[1]]

        # Find the minimum (best-fit point) of the interpolated LLR
        min_index = np.argmin(rescaled_log_r_interp)
        central_value = x_interp[min_index]

        # Print the confidence limits and central value in the desired format
        print(f"Confidence limits at threshold {threshold}: [{lower_limit:.3f}, {upper_limit:.3f}]")

def plot_llr_histo(config,observable):
    
    plt.figure(figsize=(7,6))
    
    load_dir = f"{config['main_dir']}/llr_fits_hist/{observable}{config['sample_name']}/range_{config['limits']['grid_ranges']}_resolutions_{config['limits']['grid_resolutions']}"

    data = np.load(f"{load_dir}/data_2D_Ricardos_binning_2D_case.npz")

    parameter_grid = data['parameter_grid']
    llr_kin = data['llr_kin']
    llr_rate = data['llr_rate']
    index_best_point = data['index_best_point']

    # Define interpolation points
    num_points = len(parameter_grid)
    x_interp = np.linspace(-1.2, 1.2, num_points)  # Adjusted range and density


    plt.figure(figsize=(7, 6))

    rescaled_log_r = llr_kin+llr_rate
    rescaled_log_r = -2.0*(rescaled_log_r[:] - rescaled_log_r[index_best_point])    


    parameter_grid = np.squeeze(parameter_grid)
    
    # Perform quadratic spline interpolation
    spline = UnivariateSpline(parameter_grid, rescaled_log_r, k=2, s=0)
    rescaled_log_r_interp = spline(parameter_grid)
    

    min_index = np.argmin(rescaled_log_r_interp)
    central_value = x_interp[min_index]

    # Print the results
    print(f"Central value (minimum of the LLR): {central_value:.3f}")


    lw = 1.5
    # Fill the region between the interpolated log-likelihood ± interpolated standard deviation
    color = 'indigo'
    

    plt.plot(x_interp, rescaled_log_r_interp ,linewidth=lw, color=color)
    
        
    
    # # Add horizontal lines and labels for CL
    plt.axhline(y=1.0, color='dimgray', linestyle='-.', linewidth=1.2)
    plt.axhline(y=3.84, color='dimgray', linestyle='--', linewidth=1.2)

    # Adding text labels directly on the lines
    plt.text(-0.095, 1.0, '68% CL', color='dimgray', verticalalignment='bottom', horizontalalignment='left', fontsize=11)
    plt.text(-0.095, 3.84, '95% CL', color='dimgray', verticalalignment='bottom', horizontalalignment='left', fontsize=11)
    

    plt.xlabel(r"$c_{H\tilde{G}}$", size=14)
    plt.ylabel(r"$q(\theta)$", size=14) 
    plt.ylim(-0.02,500)
    plt.xlim(-2,2)
    luminosity_info =  r"$h \rightarrow W^+ + W^- \rightarrow \ell^+ \nu_\ell\, \ell^- \nu_\ell$,  SO"
    "\n"
    r"$\mathcal{L} = 300\, \mathrm{fb}^{-1}$"
    
    #text = title + "\n" + luminosity_info
    # plt.text(0.001, 8, text,
    #      fontsize=12,   # Center the text horizontally
    #      bbox=dict(facecolor='white', edgecolor='none', alpha=0.5),horizontalalignment='center')
    
    #plt.savefig(f"{save_dir}/{fig_name}.pdf", dpi=600,bbox_inches='tight')
    
     # Find confidence limits
    find_confidence_limits(parameter_grid, rescaled_log_r, 1.0)
    find_confidence_limits(parameter_grid, rescaled_log_r, 3.84)

config = {
    'main_dir': '/project/atlas/users/sabdadin/output',
    'sample_name': 'ggf_combined_signal',
    # 'sample_name': 'ggf_signal_plus_ttbar',
    'plot_dir': '/project/atlas/users/sabdadin/output/plots',
    'limits': {
        'lumi': 300,
        'grid_ranges': [[-3,3]],
        'grid_resolutions': [5001]
    }
}
plot_llr_histo(config,"mt_tot/pt_tot")


plt.ylim(0,5)
plt.xlim(-0.1,0.1)
plt.title(rf"$m^T_{{\mathrm{{tot}}}}\otimes p^T_{{\mathrm{{tot}}}}$ Signal without Background with {config['limits']['grid_resolutions'][0]} grid_resolutions",size=14)
# plt.title(rf"$\Delta \Phi_{{jj}}\otimes \Delta \Phi_{{ll}}$ Signal + Background with {config['limits']['grid_resolutions'][0]} grid_resolutions",size=14)
plt.savefig(f"{config['plot_dir']}/llr_histo_mtot_ptot_{config['sample_name']}_grid_resolutions_{config['limits']['grid_resolutions'][0]}.pdf", dpi=600, bbox_inches='tight')