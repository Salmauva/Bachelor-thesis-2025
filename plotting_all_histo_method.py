import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import UnivariateSpline
import matplotlib.colors as colors
import matplotlib as mpl
# fig_name = "signal_background_histogram_method_2D_case_all_combined"
fig_name = "signal_only_histogram_method_2D_case_all_combined"
plot_dir = '/project/atlas/users/sabdadin/output/plots'

#1d dphi_jj
load_dir_S_dphijj = "/project/atlas/users/sabdadin/output/llr_fits_hist/dphi_jj/ggf_combined_signal/range_[[-2, 2]]_resolutions_[5000]/data_1D_Ricardos_binning"
load_dir_SB_dphijj = "/project/atlas/users/sabdadin/output/llr_fits_hist/dphi_jj/ggf_signal_plus_ttbar/range_[[-2, 2]]_resolutions_[5000]/data_1D_Ricardos_binning"
data_SB_dphijj = np.load(f"{load_dir_SB_dphijj}.npz")
data_S_dphijj= np.load(f"{load_dir_S_dphijj}.npz")
parameter_grid_SB_dphijj = np.squeeze(data_SB_dphijj['parameter_grid'])
llr_kin_SB_dphijj = data_SB_dphijj['llr_kin']
llr_rate_SB_dphijj = data_SB_dphijj['llr_rate']
index_best_point_SB_dphijj = data_SB_dphijj['index_best_point']

parameter_grid_S_dphijj = np.squeeze(data_S_dphijj['parameter_grid'])
llr_kin_S_dphijj = data_S_dphijj['llr_kin']
llr_rate_S_dphijj = data_S_dphijj['llr_rate']
index_best_point_S_dphijj = data_S_dphijj['index_best_point']
rescaled_log_r_SB_dphijj = llr_kin_SB_dphijj + llr_rate_SB_dphijj
rescaled_log_r_SB_dphijj = -2.0 * (rescaled_log_r_SB_dphijj[:] - rescaled_log_r_SB_dphijj[index_best_point_SB_dphijj])
rescaled_log_r_S_dphijj = llr_kin_S_dphijj + llr_rate_S_dphijj
rescaled_log_r_S_dphijj = -2.0 * (rescaled_log_r_S_dphijj[:] - rescaled_log_r_S_dphijj[index_best_point_S_dphijj])
plt.plot(parameter_grid_S_dphijj, rescaled_log_r_S_dphijj, lw=1.5, color="#1b9e77", label=r"$\Delta \Phi_{{jj}}$")
#plt.plot(parameter_grid_SB_dphijj, rescaled_log_r_SB_dphijj, lw=1.5, color="blue", label=r"$\Delta \Phi_{{jj}}\otimes \Delta \Phi_{{ll}}$")

#1d mtot
load_dir_S_mtot = "/project/atlas/users/sabdadin/output/llr_fits_hist/mt_tot/ggf_combined_signal/range_[[-3, 3]]_resolutions_[5001]/data_1D_Ricardos_binning_1D_case"
load_dir_SB_mtot = "/project/atlas/users/sabdadin/output/llr_fits_hist/mt_tot/ggf_signal_plus_ttbar/range_[[-3, 3]]_resolutions_[5001]/data_1D_Ricardos_binning_1D_case"
data_SB_mtot = np.load(f"{load_dir_SB_mtot}.npz")
data_S_mtot= np.load(f"{load_dir_S_mtot}.npz")
parameter_grid_SB_mtot = np.squeeze(data_SB_mtot['parameter_grid'])
llr_kin_SB_mtot = data_SB_mtot['llr_kin']
llr_rate_SB_mtot = data_SB_mtot['llr_rate']
index_best_point_SB_mtot = data_SB_mtot['index_best_point']

parameter_grid_S_mtot = np.squeeze(data_S_mtot['parameter_grid'])
llr_kin_S_mtot = data_S_mtot['llr_kin']
llr_rate_S_mtot = data_S_mtot['llr_rate']
index_best_point_S_mtot = data_S_mtot['index_best_point']
rescaled_log_r_SB_mtot = llr_kin_SB_mtot + llr_rate_SB_mtot
rescaled_log_r_SB_mtot = -2.0 * (rescaled_log_r_SB_mtot[:] - rescaled_log_r_SB_mtot[index_best_point_SB_mtot])
rescaled_log_r_S_mtot = llr_kin_S_mtot + llr_rate_S_mtot
rescaled_log_r_S_mtot = -2.0 * (rescaled_log_r_S_mtot[:] - rescaled_log_r_S_mtot[index_best_point_S_mtot])
plt.plot(parameter_grid_S_mtot, rescaled_log_r_S_mtot, lw=1.5, color="#d95f02", label=r"$m^T_{\mathrm{tot}}$")
# plt.plot(parameter_grid_SB_mtot, rescaled_log_r_SB_mtot, lw=1.5, color="blue", label=r"$m^T_{\mathrm{{tot}}})")




######2D#################
#case1:dphi_jj and dphi_ll


load_dir_SB_case1 = "/project/atlas/users/sabdadin/output/llr_fits_hist/dphi_jj/ggf_signal_plus_ttbar/range_[[-2, 2]]_resolutions_[5000]/data_1D_Ricardos_binning_2D_case"
load_dir_S = "/project/atlas/users/sabdadin/output/llr_fits_hist/dphi_jj/ggf_combined_signal/range_[[-2, 2]]_resolutions_[5000]/data_1D_Ricardos_binning_2D_case"
data_SB_case1 = np.load(f"{load_dir_SB_case1}.npz")
data_S_case1= np.load(f"{load_dir_S}.npz")
parameter_grid_SB_case1 = np.squeeze(data_SB_case1['parameter_grid'])
llr_kin_SB_case1 = data_SB_case1['llr_kin']
llr_rate_SB_case1 = data_SB_case1['llr_rate']
index_best_point_SB_case1 = data_SB_case1['index_best_point']

parameter_grid_S_case1 = np.squeeze(data_S_case1['parameter_grid'])
llr_kin_S_case1 = data_S_case1['llr_kin']
llr_rate_S_case1 = data_S_case1['llr_rate']
index_best_point_S_case1 = data_S_case1['index_best_point']
rescaled_log_r_SB_case1 = llr_kin_SB_case1 + llr_rate_SB_case1
rescaled_log_r_SB_case1 = -2.0 * (rescaled_log_r_SB_case1[:] - rescaled_log_r_SB_case1[index_best_point_SB_case1])
rescaled_log_r_S_case1 = llr_kin_S_case1 + llr_rate_S_case1
rescaled_log_r_S_case1 = -2.0 * (rescaled_log_r_S_case1[:] - rescaled_log_r_S_case1[index_best_point_S_case1])
plt.plot(parameter_grid_S_case1, rescaled_log_r_S_case1, lw=1.5, color="#7570b3", label=r"$\Delta \Phi_{{jj}}\otimes \Delta \Phi_{{ll}}$")
# plt.plot(parameter_grid_SB_case1, rescaled_log_r_SB_case1 , lw=1.5, color="indigo", label=r"$\Delta \Phi_{{jj}}\otimes \Delta \Phi_{{ll}}$")

#case2:dphi_jj and j1_px

# load_dir_SB_case2 = "/project/atlas/users/sabdadin/output/llr_fits_hist/dphi_jj/j1_px/ggf_signal_plus_ttbar/range_[[-3, 3]]_resolutions_[5001]/data_2D_Ricardos_binning_2D_case"
# load_dir_S_case2 = "/project/atlas/users/sabdadin/output/llr_fits_hist/dphi_jj/j1_px/ggf_combined_signal/range_[[-3, 3]]_resolutions_[5001]/data_2D_Ricardos_binning_2D_case"
# data_SB_case2 = np.load(f"{load_dir_SB_case2}.npz")
# data_S_case2 = np.load(f"{load_dir_S_case2}.npz")
# parameter_grid_SB_case2 = np.squeeze(data_SB_case2['parameter_grid'])
# llr_kin_SB_case2 = data_SB_case2['llr_kin']
# llr_rate_SB_case2 = data_SB_case2['llr_rate']
# index_best_point_SB_case2 = data_SB_case2['index_best_point']
# parameter_grid_S_case2 = np.squeeze(data_S_case2['parameter_grid'])
# llr_kin_S_case2 = data_S_case2['llr_kin']
# llr_rate_S_case2 = data_S_case2['llr_rate']
# index_best_point_S_case2 = data_S_case2['index_best_point']     
# rescaled_log_r_SB_case2 = llr_kin_SB_case2 + llr_rate_SB_case2
# rescaled_log_r_SB_case2 = -2.0 * (rescaled_log_r_SB_case2[:] - rescaled_log_r_SB_case2[index_best_point_SB_case2])
# rescaled_log_r_S_case2 = llr_kin_S_case2 + llr_rate_S_case2
# rescaled_log_r_S_case2 = -2.0 * (rescaled_log_r_S_case2[:] - rescaled_log_r_S_case2[index_best_point_S_case2])
# plt.plot(parameter_grid_S_case2, rescaled_log_r_S_case2, lw=1.5, color="darkorange", label="SO_dphi_jj/j1_px")
# plt.plot(parameter_grid_SB_case2, rescaled_log_r_SB_case2, lw=1.5, color="yellow", label="SB_dphi_jj/j1_px")

#case3:dphi_jj and phi_j1
# load_dir_SB_case3 = "/project/atlas/users/sabdadin/output/llr_fits_hist/dphi_jj/phi_j1/ggf_signal_plus_ttbar/range_[[-3, 3]]_resolutions_[5001]/data_2D_Ricardos_binning_2D_case"
# load_dir_S_case3 = "/project/atlas/users/sabdadin/output/llr_fits_hist/dphi_jj/phi_j1/ggf_combined_signal/range_[[-3, 3]]_resolutions_[5001]/data_2D_Ricardos_binning_2D_case"
# data_SB_case3 = np.load(f"{load_dir_SB_case3}.npz")
# data_S_case3 = np.load(f"{load_dir_S_case3}.npz")
# parameter_grid_SB_case3 = np.squeeze(data_SB_case3['parameter_grid'])
# llr_kin_SB_case3 = data_SB_case3['llr_kin']
# llr_rate_SB_case3 = data_SB_case3['llr_rate']
# index_best_point_SB_case3 = data_SB_case3['index_best_point']
# parameter_grid_S_case3 = np.squeeze(data_S_case3['parameter_grid'])
# llr_kin_S_case3 = data_S_case3['llr_kin']
# llr_rate_S_case3 = data_S_case3['llr_rate']
# index_best_point_S_case3 = data_S_case3['index_best_point']
# rescaled_log_r_SB_case3 = llr_kin_SB_case3 + llr_rate_SB_case3
# rescaled_log_r_SB_case3 = -2.0 * (rescaled_log_r_SB_case3[:] - rescaled_log_r_SB_case3[index_best_point_SB_case3])
# rescaled_log_r_S_case3 = llr_kin_S_case3 + llr_rate_S_case3
# rescaled_log_r_S_case3 = -2.0 * (rescaled_log_r_S_case3[:] - rescaled_log_r_S_case3[index_best_point_S_case3])
# plt.plot(parameter_grid_S_case3, rescaled_log_r_S_case3, lw=1.5, color="green", label="SO_dphi_jj/phi_j1")
# plt.plot(parameter_grid_SB_case3, rescaled_log_r_SB_case3, lw=1.5, color="blue", label="SB_dphi_jj/phi_j1")

#case4:dphi_jj and phi_l2
# load_dir_SB_case4 = "/project/atlas/users/sabdadin/output/llr_fits_hist/dphi_jj/phi_l2/ggf_signal_plus_ttbar/range_[[-3, 3]]_resolutions_[5001]/data_2D_Ricardos_binning_2D_case"   
# load_dir_S_case4 = "/project/atlas/users/sabdadin/output/llr_fits_hist/dphi_jj/phi_l2/ggf_combined_signal/range_[[-3, 3]]_resolutions_[5001]/data_2D_Ricardos_binning_2D_case"
# data_SB_case4 = np.load(f"{load_dir_SB_case4}.npz")
# data_S_case4 = np.load(f"{load_dir_S_case4}.npz")
# parameter_grid_SB_case4 = np.squeeze(data_SB_case4['parameter_grid'])
# llr_kin_SB_case4 = data_SB_case4['llr_kin']
# llr_rate_SB_case4 = data_SB_case4['llr_rate']
# index_best_point_SB_case4 = data_SB_case4['index_best_point']
# parameter_grid_S_case4 = np.squeeze(data_S_case4['parameter_grid'])
# llr_kin_S_case4 = data_S_case4['llr_kin']
# llr_rate_S_case4 = data_S_case4['llr_rate']
# index_best_point_S_case4 = data_S_case4['index_best_point']
# rescaled_log_r_SB_case4 = llr_kin_SB_case4 + llr_rate_SB_case4
# rescaled_log_r_SB_case4 = -2.0 * (rescaled_log_r_SB_case4[:] - rescaled_log_r_SB_case4[index_best_point_SB_case4])
# rescaled_log_r_S_case4 = llr_kin_S_case4 + llr_rate_S_case4
# rescaled_log_r_S_case4 = -2.0 * (rescaled_log_r_S_case4[:] - rescaled_log_r_S_case4[index_best_point_S_case4])
# plt.plot(parameter_grid_S_case4, rescaled_log_r_S_case4, lw=1.5, color="red", label="SO_dphi_jj/phi_l2")
# plt.plot(parameter_grid_SB_case4, rescaled_log_r_SB_case4, lw=1.5, color="black", label="SB_dphi_jj/phi_l2")

#case5:dphi_jj and phi_j2
# load_dir_SB_case5 = "/project/atlas/users/sabdadin/output/llr_fits_hist/dphi_jj/phi_j2/ggf_signal_plus_ttbar/range_[[-3, 3]]_resolutions_[5001]/data_2D_Ricardos_binning_2D_case"
# load_dir_S_case5 = "/project/atlas/users/sabdadin/output/llr_fits_hist/dphi_jj/phi_j2/ggf_combined_signal/range_[[-3, 3]]_resolutions_[5001]/data_2D_Ricardos_binning_2D_case"
# data_SB_case5 = np.load(f"{load_dir_SB_case5}.npz")     
# data_S_case5 = np.load(f"{load_dir_S_case5}.npz")
# parameter_grid_SB_case5 = np.squeeze(data_SB_case5['parameter_grid'])
# llr_kin_SB_case5 = data_SB_case5['llr_kin']
# llr_rate_SB_case5 = data_SB_case5['llr_rate']
# index_best_point_SB_case5 = data_SB_case5['index_best_point']
# parameter_grid_S_case5 = np.squeeze(data_S_case5['parameter_grid'])
# llr_kin_S_case5 = data_S_case5['llr_kin']
# llr_rate_S_case5 = data_S_case5['llr_rate']
# index_best_point_S_case5 = data_S_case5['index_best_point']
# rescaled_log_r_SB_case5 = llr_kin_SB_case5 + llr_rate_SB_case5
# rescaled_log_r_SB_case5 = -2.0 * (rescaled_log_r_SB_case5[:] - rescaled_log_r_SB_case5[index_best_point_SB_case5])
# rescaled_log_r_S_case5 = llr_kin_S_case5 + llr_rate_S_case5
# rescaled_log_r_S_case5 = -2.0 * (rescaled_log_r_S_case5[:] - rescaled_log_r_S_case5[index_best_point_S_case5])
# plt.plot(parameter_grid_S_case5, rescaled_log_r_S_case5, lw=1.5, color="orange", label="SO_dphi_jj/phi_j2")
# plt.plot(parameter_grid_SB_case5, rescaled_log_r_SB_case5, lw=1.5, color="green", label="SB_dphi_jj/phi_j2")

#case6:dphi_jj and phi_l1
# load_dir_SB_case6 = "/project/atlas/users/sabdadin/output/llr_fits_hist/dphi_jj/phi_l1/ggf_signal_plus_ttbar/range_[[-3, 3]]_resolutions_[5001]/data_2D_Ricardos_binning_2D_case"
# load_dir_S_case6 = "/project/atlas/users/sabdadin/output/llr_fits_hist/dphi_jj/phi_l1/ggf_combined_signal/range_[[-3, 3]]_resolutions_[5001]/data_2D_Ricardos_binning_2D_case"
# data_SB_case6 = np.load(f"{load_dir_SB_case6}.npz")
# data_S_case6 = np.load(f"{load_dir_S_case6}.npz")
# parameter_grid_SB_case6 = np.squeeze(data_SB_case6['parameter_grid'])
# llr_kin_SB_case6 = data_SB_case6['llr_kin']
# llr_rate_SB_case6 = data_SB_case6['llr_rate']
# index_best_point_SB_case6 = data_SB_case6['index_best_point']
# parameter_grid_S_case6 = np.squeeze(data_S_case6['parameter_grid'])
# llr_kin_S_case6 = data_S_case6['llr_kin']
# llr_rate_S_case6 = data_S_case6['llr_rate'] 
# index_best_point_S_case6 = data_S_case6['index_best_point']
# rescaled_log_r_SB_case6 = llr_kin_SB_case6 + llr_rate_SB_case6
# rescaled_log_r_SB_case6 = -2.0 * (rescaled_log_r_SB_case6[:] - rescaled_log_r_SB_case6[index_best_point_SB_case6])
# rescaled_log_r_S_case6 = llr_kin_S_case6 + llr_rate_S_case6
# rescaled_log_r_S_case6 = -2.0 * (rescaled_log_r_S_case6[:] - rescaled_log_r_S_case6[index_best_point_S_case6])
# plt.plot(parameter_grid_S_case6, rescaled_log_r_S_case6, lw=1.5, color="purple", label="SO_dphi_jj/phi_l1")
# plt.plot(parameter_grid_SB_case6, rescaled_log_r_SB_case6, lw=1.5, color="pink", label="SB_dphi_jj/phi_l1")

#case7:mt_tot and pt_tot
load_dir_SB_case7 = "/project/atlas/users/sabdadin/output/llr_fits_hist/mt_tot/pt_totggf_signal_plus_ttbar/range_[[-3, 3]]_resolutions_[5001]/data_2D_Ricardos_binning_2D_case"
load_dir_S_case7 = "/project/atlas/users/sabdadin/output/llr_fits_hist/mt_tot/pt_totggf_combined_signal/range_[[-3, 3]]_resolutions_[5001]/data_2D_Ricardos_binning_2D_case"
data_SB_case7 = np.load(f"{load_dir_SB_case7}.npz")
data_S_case7 = np.load(f"{load_dir_S_case7}.npz")
parameter_grid_SB_case7 = np.squeeze(data_SB_case7['parameter_grid'])
llr_kin_SB_case7 = data_SB_case7['llr_kin']
llr_rate_SB_case7 = data_SB_case7['llr_rate']   
index_best_point_SB_case7 = data_SB_case7['index_best_point']
parameter_grid_S_case7 = np.squeeze(data_S_case7['parameter_grid'])
llr_kin_S_case7 = data_S_case7['llr_kin']
llr_rate_S_case7 = data_S_case7['llr_rate']
index_best_point_S_case7 = data_S_case7['index_best_point']
rescaled_log_r_SB_case7 = llr_kin_SB_case7 + llr_rate_SB_case7
rescaled_log_r_SB_case7 = -2.0 * (rescaled_log_r_SB_case7[:] - rescaled_log_r_SB_case7[index_best_point_SB_case7])
rescaled_log_r_S_case7 = llr_kin_S_case7 + llr_rate_S_case7
rescaled_log_r_S_case7 = -2.0 * (rescaled_log_r_S_case7[:] - rescaled_log_r_S_case7[index_best_point_S_case7])
plt.plot(parameter_grid_S_case7, rescaled_log_r_S_case7, lw=1.5, color="#e7298a", label=r"$m^T_{{\mathrm{{tot}}}}\otimes p^T_{{\mathrm{{tot}}}}$")
# plt.plot(parameter_grid_SB_case7, rescaled_log_r_SB_case7, lw=1.5, color="brown", label=r"$m^T_{{\mathrm{{tot}}}}\otimes p^T_{{\mathrm{{tot}}}}$")



# # Add horizontal lines and labels for CL
plt.axhline(y=1.0, color='dimgray', linestyle='-.', linewidth=1.2)
plt.axhline(y=3.84, color='dimgray', linestyle='--', linewidth=1.2)

# Adding text labels directly on the lines
plt.text(-0.01, 1.0, '68% CL', color='dimgray', verticalalignment='bottom', horizontalalignment='left', fontsize=11)
plt.text(-0.01, 3.84, '95% CL', color='dimgray', verticalalignment='bottom', horizontalalignment='left', fontsize=11)
plt.xlabel(r"$c_{H\tilde{G}}$", size=14)
plt.ylabel(r"$q(\theta)$", size=14)
luminosity_info = (
    r"$h \rightarrow W^+ + W^- \rightarrow \ell^+ \nu_\ell\, \ell^- \nu_\ell$,  SO"
    "\n"
    r"$\mathcal{L} = 300\, \mathrm{fb}^{-1}$"
)

plt.text(-0.003, 8, luminosity_info,
         fontsize=12,   # Center the text horizontally
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.5),horizontalalignment='center')
plt.legend(title_fontsize=12, frameon=False,fontsize=11        ,handlelength=2.5,                         # Adjust length of legend lines
        labelspacing=0.4,facecolor='white',loc = "upper center",bbox_to_anchor=(0.47, 0.8))
plt.ylim(0, 10)
plt.xlim(-0.05, 0.05)
plt.savefig(f"{plot_dir}/{fig_name}.pdf", dpi=600, bbox_inches='tight')

