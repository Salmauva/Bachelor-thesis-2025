# import os
# import logging
# import argparse 
# import yaml
# from madminer import MadMiner
# from madminer.plotting import plot_1d_morphing_basis

# # MadMiner output
# logging.basicConfig(
#     format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
#     datefmt='%H:%M',
#     level=logging.INFO
# )

# # Output of all other modules (e.g. matplotlib)
# for key in logging.Logger.manager.loggerDict:
#     if "madminer" not in key:
#         logging.getLogger(key).setLevel(logging.WARNING)


# def setup_madminer(main_dir,plot_dir):
#     """
#     Sets up the MadMiner instance for WH signal, with only the CP-odd operator (oHWtil), and morphing up to 2nd order.
#     """

#     # Instance of MadMiner core class
#     miner = MadMiner()

#     miner.add_parameter(
#         lha_block='smeftcpv',
#         lha_id=4,
#         parameter_name='cHGtil',
#         morphing_max_power=2, # interference + squared terms
#         parameter_range=(-1.2,1.2),
#         param_card_transform='1.0*theta' # mandatory to avoid a crash due to a bug
#     )

#     # Only want the SM benchmark specifically - let Madminer choose the others
#     miner.add_benchmark({'cHGtil':0.00},'sm')
#     miner.add_benchmark({'cHGtil':1.15},'pos_chgtil')
#     miner.add_benchmark({'cHGtil':-1.035},'neg_chgtil')

#     # Morphing - automatic optimization to avoid large weights
#     miner.set_morphing(max_overall_power=2,include_existing_benchmarks=True)

#     miner.save(f'{main_dir}/setup_1D_CP_odd.h5')

#     morphing_basis=plot_1d_morphing_basis(miner.morpher,xlabel=r'$\tilde{c_{HG}}$',xrange=(-1.2,1.2))
#     morphing_basis.savefig(f'{plot_dir}/morphing_basis_1D_CP_odd.pdf')

#     return miner
    

# if __name__ == "__main__":

#     parser = argparse.ArgumentParser(description='Creates MadMiner parameter and morphing setup file for a WH signal, with only the CP-odd operator (oHWtil), \
#                                morphing up to second order (SM + SM-EFT interference + EFT^2 term).',
#                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#     parser.add_argument('--config_file', help='Path to the YAML configuration file', default='config_1D_CP_odd.yaml')
#     args = parser.parse_args()

#     # Read main_dir and plot_dir from the YAML configuration file
#     with open(args.config_file, 'r') as config_file:
#         config = yaml.safe_load(config_file)
#         main_dir = config['main_dir']
#         plot_dir = config['plot_dir']

#     # MadMiner setup function
#     setup_madminer(main_dir, plot_dir)
#############################################
# import logging
# from madminer import MadMiner
# from madminer.plotting import plot_1d_morphing_basis  # <-- Fixes your import error

# logging.basicConfig(level=logging.INFO)

# miner = MadMiner()

# miner.add_parameter(
#     lha_block='smeftcpv',
#     lha_id=4,
#     parameter_name='cHGtil',
#     morphing_max_power=2,
#     param_card_transform='1.0*theta',
#     parameter_range=(-1.0, 1.0)
# )

# miner.add_benchmark({'cHGtil': 0.00}, 'sm')
# miner.add_benchmark({'cHGtil': 1.15}, 'pos_chgtil')
# miner.add_benchmark({'cHGtil': -1.035}, 'neg_chgtil')
# miner.set_morphing(include_existing_benchmarks=True, max_overall_power=2)

# fig = plot_1d_morphing_basis(
#     miner.morpher,
#     xlabel=r'$\tilde{c_{HG}}$',
#     xrange=(-1.2, 1.2)
# )
# fig.savefig('plots/morphing_basis_1D_CP_odd.pdf')

# miner.save('output/setup_1D_CP_odd.h5')
import os
import logging
import argparse 
import yaml
from madminer import MadMiner
from madminer.plotting import plot_1d_morphing_basis

# MadMiner output
logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.INFO
)

# Output of all other modules (e.g. matplotlib)
for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
        logging.getLogger(key).setLevel(logging.WARNING)


def setup_madminer(main_dir,plot_dir):
    """
    Sets up the MadMiner instance for WH signal, with only the CP-odd operator (oHWtil), and morphing up to 2nd order.
    """

    # Instance of MadMiner core class
    miner = MadMiner()

    miner.add_parameter(
        lha_block='smeftcpv',
        lha_id=3,
        parameter_name='cHGtil',
        morphing_max_power=2, # interference + squared terms
        parameter_range=(-3,3),
        param_card_transform='1.0*theta' # mandatory to avoid a crash due to a bug
    )

    miner.add_benchmark({'cHGtil':0.00},'sm')
    miner.add_benchmark({'cHGtil':2.868},'pos_chgtil')
    miner.add_benchmark({'cHGtil':-2.972},'neg_chgtil')

    # Morphing - automatic optimization to avoid large weights
    miner.set_morphing(max_overall_power=2,include_existing_benchmarks=True,n_trials = 1000, n_test_thetas=1000)

    miner.save(f'{main_dir}/setup_1D_cHGtil.h5')

    morphing_basis=plot_1d_morphing_basis(miner.morpher,xlabel=r'$\tilde{c_{HG}}$',xrange=(-3,3))
    morphing_basis.savefig(f'{plot_dir}/morphing_basis_1D_cHGtil.pdf')

    return miner
    
 
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Creates MadMiner parameter and morphing setup file for a WH signal, with only the CP-odd operator (oHGtil), \
                               morphing up to second order (SM + SM-EFT interference + EFT^2 term).',
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config_file', help='Path to the YAML configuration file', default='config_1D_cHGtil.yaml')
    args = parser.parse_args()

    # Read main_dir and plot_dir from the YAML configuration file
    with open(args.config_file, 'r') as config_file:
        config = yaml.safe_load(config_file)
        main_dir = config['main_dir']
        plot_dir = config['plot_dir']

    # MadMiner setup function
    setup_madminer(main_dir, plot_dir)