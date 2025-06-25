# #use madminer tutorials
# import h5py
# # import matplotlib
# # matplotlib.use('TkAgg')  # Or 'Qt5Agg' if TkAgg is unavailable
# import matplotlib.pyplot as plt
# import os
# from matplotlib.backends.backend_pdf import PdfPages
# import numpy as np
# from madminer.plotting import plot_distributions
# _ = plot_distributions(
#     filename='/project/atlas/users/sabdadin/output/ggf_signal_plus_ttbar.h5',
#     observables=['dphi_jj', 'dphi_ll', 'mt_tot', 'pt_tot'],
#     parameter_points=  [  np.array([0.0]),      # Standard Model
#     np.array([0.1]),   # Positive CP-odd operator
#     np.array([-0.1]),  # Negative CP-odd operator
# ],
#     line_labels= [
#     'SM (cHGtil = 0.0)',
#     'pos_chgtil (cHGtil = 0.1)',
#     'neg_chgtil (cHGtil = -0.1)'
# ],
#     uncertainties='none',
#     n_bins=20,
#     n_cols=3,
#     normalize=True,
#     sample_only_from_closest_benchmark=True
# )
# # Modify x-axis labels
# fig = plt.gcf()
# axes = fig.get_axes()
# custom_labels = [
#     r'$\Delta\phi_{jj}$ (rad)',
#     r'$\Delta\phi_{\ell\ell}$ (rad)',
#     r'$m_T^\mathrm{tot}$ (GeV)'
#     r'$p_T^\mathrm{tot}$ (GeV)'
# ]
# for ax, label in zip(axes, custom_labels):
#     ax.set_xlabel(label)
# output_pdf = '/project/atlas/users/sabdadin/output/plots/plotsnew.pdf'
# plt.savefig(output_pdf)
# print(f"Plots saved to {output_pdf}")

#plot SM+background vs BSM
#use madminer tutorials
import h5py
# import matplotlib
# matplotlib.use('TkAgg')  # Or 'Qt5Agg' if TkAgg is unavailable
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from madminer.plotting import plot_distributions
_ = plot_distributions(
    filename='/project/atlas/users/sabdadin/output/ggf_signal_plus_ttbar.h5',
    observables=['dphi_jj', 'dphi_ll', 'mt_tot', 'pt_tot'],

    parameter_points=  [  np.array([0.0]),      # Standard Model
    np.array([1.5]),   # Positive CP-odd operator
    np.array([-1.5]),  # Negative CP-odd operator
],
    line_labels= [
    'SM + Background',
    r'$C_{H \tilde{G}} =1.5$',
    r'$C_{H \tilde{G}} = -1.5$'
],
    uncertainties='none',
    n_bins=20,
    n_cols=3,
    normalize=True,
    sample_only_from_closest_benchmark=True
)
# Modify x-axis labels
fig = plt.gcf()
axes = fig.get_axes()
custom_labels = [
    r'$\Delta\phi_{jj}$ (rad)',
    r'$\Delta\phi_{\ell\ell}$ (rad)',
    r'$m_T^\mathrm{tot}$ (GeV)',
    r'$p_T^\mathrm{tot}$ (GeV)'
]
for ax, label in zip(axes, custom_labels):
    ax.set_xlabel(label)
ax.set_ylabel(r'Normalized events')
output_pdf = '/project/atlas/users/sabdadin/output/plots/plotsnew_small_benchmarks_plus_background_final_1.5.pdf'
plt.savefig(output_pdf)
print(f"Plots saved to {output_pdf}")

#plot SM without background vs BSM
_ = plot_distributions(
    filename='/project/atlas/users/sabdadin/output/ggf_combined_signal.h5',
    observables=['dphi_jj', 'dphi_ll', 'mt_tot', 'pt_tot'],
    parameter_points=  [  np.array([0.0]),      # Standard Model
    np.array([1.5]),   # Positive CP-odd operator
    np.array([-1.5]),  # Negative CP-odd operator
],
    line_labels= [
    'SM',
    r'$C_{H \tilde{G}} = 1.5$',
    r'$C_{H \tilde{G}} = -1.5$'
],
    uncertainties='none',
    n_bins=20,
    n_cols=3,
    normalize=True,
    sample_only_from_closest_benchmark=True
)
# Modify x-axis labels
fig = plt.gcf()
axes = fig.get_axes()
custom_labels = [
    r'$\Delta\phi_{jj}$ (rad)',
    r'$\Delta\phi_{\ell\ell}$ (rad)',
    r'$m_T^\mathrm{tot}$ (GeV)',
    r'$p_T^\mathrm{tot}$ (GeV)'
]
for ax, label in zip(axes, custom_labels):
    ax.set_xlabel(label)
    ax.set_ylabel(r'Normalized events')
output_pdf = '/project/atlas/users/sabdadin/output/plots/plotsnew_small_benchmarks_without_background_final_1.5.pdf'
plt.savefig(output_pdf)
print(f"Plots saved to {output_pdf}")