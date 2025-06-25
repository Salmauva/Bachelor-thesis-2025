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
       


def plot_llr_individual(config):

    plt.figure(figsize=(7,6))

    load_dir = f"{config['main_dir']}/llr_fits/{config['limits']['prior']}/{config['limits']['observables']}/{config['limits']['model']}/{config['sample_name']}/range_{config['limits']['grid_ranges']}_resolutions_{config['limits']['grid_resolutions']}"
    fig_name = f"llr_fit_individual_{config['sample_name']}_range_{config['limits']['grid_ranges']}_resolutions_{config['limits']['grid_resolutions']}"

    os.makedirs(f"{config['plot_dir']}/llr_fits/{config['limits']['prior']}/{config['limits']['observables']}/{config['limits']['model']}/",exist_ok=True)
    save_dir = f"{config['plot_dir']}/llr_fits/{config['limits']['prior']}/{config['limits']['observables']}/{config['limits']['model']}/"
    
    colors = ['#332288', '#117733', '#CC6677', '#882255','44AA99']

    # Define interpolation points
    num_points = 1001
    x_interp = np.linspace(-1.2, 1.2, num_points)  # Adjusted range and density



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

      if config['limits']['method'] == 'sally':
         title = r'$\bf{SALLY}$' 

      if config['limits']['method'] == 'alices':
         title = r'$\bf{ALICES}$' + r'$\ (\alpha = 5, n_{\theta_0} = 1000)$'

      if config['limits']['method'] == 'alice':
         title = r'$\bf{ALICE}$' + r'$\ (\alpha = 0, n_{\theta_0} = 1000)$'

      # Perform quadratic spline interpolation
      spline = UnivariateSpline(parameter_grid, rescaled_log_r, k=2, s=0)
      rescaled_log_r_interp = spline(x_interp)

      

      lw=1.5
      plt.plot(x_interp, rescaled_log_r_interp ,linewidth=lw, label=f"Estimator {estimator_number}",color=colors[i])    
      
      # Add horizontal lines and labels for CL
      plt.axhline(y=1.0, color='gray', linestyle='-.', linewidth=1.2)
      plt.axhline(y=3.84, color='gray', linestyle='--', linewidth=1.2)

      # Adding text labels directly on the lines
      plt.text(-0.095, 1.0, '68% CL', color='gray', verticalalignment='bottom', horizontalalignment='left', fontsize=11)
      plt.text(-0.095, 3.84, '95% CL', color='gray', verticalalignment='bottom', horizontalalignment='left', fontsize=11)

    plt.xlabel(r"$c_{H\tilde{W}}$", size=14)
    plt.ylabel(r"$q(\theta)$", size=14) 
    plt.ylim(-0.02,10)
    plt.xlim(-0.1,0.1)
    luminosity_info =  r"$u \bar{d} \rightarrow W^+ H \rightarrow \mu^+ \nu_{\mu} b \bar{b}$, KO" + "\n" + r"$\mathcal{L} = 300\, \mathrm{fb}^{-1}$"  
    
    text = title + "\n" + luminosity_info
    plt.text(-0.003, 8, text,
         fontsize=12,   # Center the text horizontally
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.5),horizontalalignment='center')
    plt.legend(title_fontsize=12, frameon=False,fontsize=11        ,handlelength=2.5,                         # Adjust length of legend lines
        labelspacing=0.4,facecolor='white',loc = "upper center",bbox_to_anchor=(0.47, 0.8))
    
    plt.savefig(f"{save_dir}/{fig_name}.pdf")
    

def calculate_std(config):
    
    load_dir = f"{config['main_dir']}/llr_fits/{config['limits']['prior']}/{config['limits']['observables']}/{config['limits']['model']}/{config['sample_name']}/range_{config['limits']['grid_ranges']}_resolutions_{config['limits']['grid_resolutions']}"
    

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

def plot_llr_ensemble(config):
    
    plt.figure(figsize=(7,6))
    
    load_dir = f"{config['main_dir']}/llr_fits/{config['limits']['prior']}/{config['limits']['observables']}/{config['limits']['model']}/{config['sample_name']}/range_{config['limits']['grid_ranges']}_resolutions_{config['limits']['grid_resolutions']}"

    fig_name = f"llr_fit_ensemble_{config['sample_name']}_range_{config['limits']['grid_ranges']}_resolutions_{config['limits']['grid_resolutions']}"

    os.makedirs(f"{config['plot_dir']}/llr_fits/{config['limits']['prior']}/{config['limits']['observables']}/{config['limits']['model']}/",exist_ok=True)
    save_dir = f"{config['plot_dir']}/llr_fits/{config['limits']['prior']}/{config['limits']['observables']}/{config['limits']['model']}/"
    
    data = np.load(f"{load_dir}/ensemble_data.npz")

    parameter_grid = data['parameter_grid']
    llr_kin = data['llr_kin']
    llr_rate = data['llr_rate']
    index_best_point = data['index_best_point']

    # Define interpolation points
    num_points = 1001
    x_interp = np.linspace(-1.2, 1.2, num_points)  # Adjusted range and density


    if config['limits']['method'] == 'sally':
      title = r'$\bf{SALLY}$' 
      color = 'mediumvioletred'

    if config['limits']['method'] == 'alices':
      title = r'$\bf{ALICES}$' + r'$\ (\alpha = 5, n_{\theta_0} = 10000)$'
      color = 'mediumblue'

    if config['limits']['method'] == 'alice':
      title = title = r'$\bf{ALICE}$' + r'$\ (\alpha = 0, n_{\theta_0} = 1000)$'
      color = 'darkgreen'
        


    plt.figure(figsize=(7, 6))

    rescaled_log_r = llr_kin+llr_rate
    rescaled_log_r = -2.0*(rescaled_log_r[:] - rescaled_log_r[index_best_point])    

    std = calculate_std(config)
    std = np.squeeze(std)
    parameter_grid = np.squeeze(parameter_grid)
    
    # Perform quadratic spline interpolation
    spline = UnivariateSpline(parameter_grid, rescaled_log_r, k=2, s=0)
    rescaled_log_r_interp = spline(x_interp)
    
    # Perform quadratic spline interpolation for the standard deviation
    spline_std = UnivariateSpline(parameter_grid, std, k=2, s=0)
    std_interp = spline_std(x_interp)
    
    min_index = np.argmin(rescaled_log_r_interp)
    central_value = x_interp[min_index]
    std_at_min = std_interp[min_index]  # Standard deviation at the minimum
    
    # Print the results
    print(f"Central value (minimum of the LLR): {central_value:.3f}")
    print(f"Standard deviation at the central value: {std_at_min:.3f}")
    

    lw = 1.5
    # Fill the region between the interpolated log-likelihood ± interpolated standard deviation
    plt.fill_between(x_interp, rescaled_log_r_interp - std_interp, rescaled_log_r_interp + std_interp, color=color, alpha=0.1)
    

    plt.plot(x_interp, rescaled_log_r_interp ,linewidth=lw, color=color)    
    
    # # Add horizontal lines and labels for CL
    # plt.axhline(y=1.0, color='dimgray', linestyle='-.', linewidth=1.2)
    # plt.axhline(y=3.84, color='dimgray', linestyle='--', linewidth=1.2)

    # Adding text labels directly on the lines
    # plt.text(-0.095, 1.0, '68% CL', color='dimgray', verticalalignment='bottom', horizontalalignment='left', fontsize=11)
    # plt.text(-0.095, 3.84, '95% CL', color='dimgray', verticalalignment='bottom', horizontalalignment='left', fontsize=11)
    

    plt.xlabel(r"$c_{H\tilde{W}}$", size=14)
    plt.ylabel(r"$q(\theta)$", size=14) 
    plt.ylim(-0.02,500)
    plt.xlim(-1.2,1.2)
    luminosity_info =  r"$u \bar{d} \rightarrow W^+ H \rightarrow \mu^+ \nu_{\mu} b \bar{b}$, KO" + "\n" + r"$\mathcal{L} = 300\, \mathrm{fb}^{-1}$"  
    
    #text = title + "\n" + luminosity_info
    # plt.text(0.001, 8, text,
    #      fontsize=12,   # Center the text horizontally
    #      bbox=dict(facecolor='white', edgecolor='none', alpha=0.5),horizontalalignment='center')
    
    #plt.savefig(f"{save_dir}/{fig_name}.pdf", dpi=600,bbox_inches='tight')
    
     # Find confidence limits
    find_confidence_limits(x_interp, rescaled_log_r_interp, 1.0)
    find_confidence_limits(x_interp, rescaled_log_r_interp, 3.84)

config = {
    'main_dir': '/project/atlas/users/sabdadin/output',
    'sample_name': 'ggf_signal_plus_ttbar',
    'plot_dir': '/project/atlas/users/sabdadin/output/plots',
    'limits': {
        'mode': 'ml',
        'observables': 'kinematic_only',
        'prior': 'alices_uniform_prior_-0.5_0.5_10000_thetas_10e7_samples_CP_odd', #'alices_gaussian_prior_0_1.2_10000_thetas_10e7_samples_CP_odd',
        'model': 'alices_hidden_[100, 100]_tanh_alpha_1_epochs_50_bs_128',
        'method': 'alices',
        'lumi': 300,
        'grid_ranges': [-0.5, 0.5], #[-0.2, 0.2],
        'grid_resolutions': 801
        ,'nestimators': 5,
    }
}
    
plot_llr_ensemble(config)