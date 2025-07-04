import logging
import os
import numpy as np
from madminer.limits import AsymptoticLimits

# Configure logging
logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.DEBUG
)

# Set CUDA environment variables
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "0")

def evaluate_limits_histo(config):
    """Evaluates and saves the log likelihood ratio for each estimator in the ensemble"""

    filename = f"{config['main_dir']}/{config['sample_name']}.h5"
    vars = ['mt_tot']

    limits_file = AsymptoticLimits(filename)

    parameter_grid, p_values, index_best_point, llr_kin, llr_rate, (histos, observed, observed_weights) = limits_file.expected_limits(
        mode='histo',
        theta_true=[0.0],
        include_xsec=not config['limits']['shape_only'],
        hist_vars=vars,
        luminosity=config['limits']['lumi'] * 1000.0,
        return_asimov=True,
        test_split=1.0,
        n_histo_toys=None,
        grid_ranges=config['limits']['grid_ranges'],hist_bins = [[0,400,800,1e9]],
        grid_resolutions=config['limits']['grid_resolutions']
    )

    save_dir = f"{config['main_dir']}/llr_fits_hist/{vars[0]}/{config['sample_name']}/range_{config['limits']['grid_ranges']}_resolutions_{config['limits']['grid_resolutions']}"
    os.makedirs(save_dir, exist_ok=True)

    np.savez(f"{save_dir}/data_1D_Ricardos_binning_1D_case.npz", parameter_grid=parameter_grid, p_values=p_values,
             index_best_point=index_best_point, llr_kin=llr_kin, llr_rate=llr_rate)

if __name__ == "__main__":
    config = {
        'main_dir': '/project/atlas/users/sabdadin/output',
        # 'sample_name': 'ggf_combined_signal',
        'sample_name': 'ggf_signal_plus_ttbar',
        'limits': {
            'lumi': 300,
            'grid_ranges': [[-3, 3]],
            'grid_resolutions': [5001],
            'shape_only': False  # This needs to be defined to avoid KeyError
        }
    }

    evaluate_limits_histo(config)