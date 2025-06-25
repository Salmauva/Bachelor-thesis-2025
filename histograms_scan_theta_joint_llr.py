import numpy as np
import logging
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import os
from madminer.ml import ParameterizedRatioEstimator,ScoreEstimator, Ensemble
from scipy.stats import gaussian_kde
from madminer.ml import  Ensemble
from madminer.plotting import plot_histograms
from madminer.sampling import SampleAugmenter
from madminer.limits import AsymptoticLimits
import logging
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from madminer.sampling import SampleAugmenter
from madminer import sampling
from madminer.ml import ParameterizedRatioEstimator
import logging
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from madminer.sampling import SampleAugmenter
from madminer import sampling
from madminer.ml import ParameterizedRatioEstimator

logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.INFO
)



# Output of all other modules (e.g. matplotlib)
for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
        logging.getLogger(key).setLevel(logging.WARNING)

# Choose the GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "0")


############no_joint_llr##################
#model_path = "/project/atlas/users/sabdadin/output/models/alices_uniform_prior_-0.5_0.5_10000_thetas_10e7_samples_CP_odd/kinematic_only/alices_hidden_[100, 100]_tanh_alpha_0_epochs_50_bs_128/alices_ensemble_ggf_signal_plus_ttbar" '/project/atlas/users/sabdadin/output/models/alices_gaussian_prior_0_1.2_10000_thetas_10e7_samples_CP_odd/kinematic_only/alices_hidden_[100, 100]_tanh_alpha_1_epochs_100_bs_128/alices_ensemble_ggf_signal_plus_ttbar'
# model_path = '/project/atlas/users/sabdadin/output/models/alices_gaussian_prior_0_1.2_10000_thetas_10e7_samples_CP_odd/kinematic_only/alices_hidden_[100, 100]_tanh_alpha_0_epochs_50_bs_128/alices_ensemble_ggf_signal_plus_ttbar'
# alices = Ensemble()
# alices.load(model_path)
# #theta_each = np.linspace(-1.0,1.0,50)
# theta_each = np.linspace(-2.972,2.868,50)
# theta_grid = np.array([theta_each]).T

# log_r_hat, _ = alices.evaluate_log_likelihood_ratio(
#     theta=theta_grid,
#     x= '/project/atlas/users/sabdadin/output/training_samples/alices_alices_gaussian_prior_0_1.2_10000_thetas_10e7_samples_CP_odd/x_train_ratio_ggf_signal_plus_ttbar_0.npy', #'/project/atlas/users/sabdadin/output/training_samples/alices_alices_uniform_prior_-0.5_0.5_10000_thetas_10e7_samples_CP_odd/x_train_ratio_ggf_signal_plus_ttbar_0.npy',
#     evaluate_score=False,
#     test_all_combinations = True
# )
# expected_llr =log_r_hat
# expected_llr = np.mean(log_r_hat,axis=0)

# # Create histogram
# #plt.hist(log_r_hat, bins=30, edgecolor='black', density=True, alpha=0.7)
# bins = np.linspace(-0.03,0.03,30)
# plt.hist(expected_llr, bins=bins, histtype='step', color='teal', density=True,linewidth=2)
# plt.hist(expected_llr, bins=bins, color='teal', density=True, alpha=0.2)
# plt.plot(-0.5,0.5,label = "Signal Only", linewidth = 2, color='teal')

# model_path = '/project/atlas/users/sabdadin/output/models/alices_gaussian_prior_0_1.2_10000_thetas_10e7_samples_CP_odd/kinematic_only/alices_hidden_[50]_tanh_alpha_0_epochs_50_bs_128/alices_ensemble_ggf_combined_signal'
# alices = Ensemble()
# alices.load(model_path)
# #theta_each = np.linspace(-1.0,1.0,50)
# theta_each = np.linspace(-2.972,2.868,50)
# theta_grid = np.array([theta_each]).T

# log_r_hat, _ = alices.evaluate_log_likelihood_ratio(
#     theta=theta_grid,
#     x='/project/atlas/users/sabdadin/output/training_samples/alices_alices_gaussian_prior_0_1.2_10000_thetas_10e7_samples_CP_odd/x_train_ratio_ggf_combined_signal_0.npy',
#     evaluate_score=False,
#     test_all_combinations = True
# )
# expected_llr =log_r_hat
# expected_llr = np.mean(log_r_hat,axis=0)



# bins = np.linspace(-0.3,0.3,30)
# plt.hist(expected_llr, bins=bins, histtype='step', color='mediumblue', density=True,linewidth=2)
# plt.hist(expected_llr, bins=bins, color='mediumblue', density=True, alpha=0.2)
# plt.plot(-0.5,0.5,label = "Signal + Backgrounds", linewidth = 2, color = "mediumblue")
# plt.xlim(-0.03,0.03)
# # plt.xlabel(r'$r(x,z)$')
# plt.xlabel(r'$\log \ \hat{r}(x|\theta_0,\theta_1)$', size=14)
# plt.ylabel('Normalized distribution', size=14)
# plt.legend(frameon=False, fontsize=11)
# plt.savefig("llr_hist_cp_odd_alpha0_default_theta_range_flat_prior.pdf", dpi=600,bbox_inches='tight')

################ALICES_joint_llr########################################  
plt.figure(figsize=(10, 6))            
thetas = np.load('/project/atlas/users/sabdadin/output/training_samples/alices_alices_gaussian_prior_0_1.2_10000_thetas_10e7_samples_CP_odd/theta0_train_ratio_ggf_combined_signal_0.npy') 
joint_llr = np.load('/project/atlas/users/sabdadin/output/training_samples/alices_alices_gaussian_prior_0_1.2_10000_thetas_10e7_samples_CP_odd/r_xz_train_ratio_ggf_combined_signal_0.npy') 

filtered_indices =  np.where((thetas >= -0.15) & (thetas <= 0.15))

filtered_joint_llr = joint_llr[filtered_indices]

log_filtered_joint_llr=np.log(joint_llr)

bins = np.linspace(-0.15, 0.15,50) 
plt.hist(log_filtered_joint_llr, bins=bins, histtype='step', color='teal', density=True,linewidth=2)
plt.hist(log_filtered_joint_llr, bins=bins, color='teal', density=True, alpha=0.2)
plt.plot(-0.5,0.5,label = "Signal Only", linewidth = 2, color = "teal")


thetas = np.load('/project/atlas/users/sabdadin/output/training_samples/alices_alices_gaussian_prior_0_1.2_10000_thetas_10e7_samples_CP_odd/theta0_train_ratio_ggf_signal_plus_ttbar_0.npy') 
joint_llr = np.load('/project/atlas/users/sabdadin/output/training_samples/alices_alices_gaussian_prior_0_1.2_10000_thetas_10e7_samples_CP_odd/r_xz_train_ratio_ggf_signal_plus_ttbar_0.npy') 

filtered_indices =  np.where((thetas >= -0.15) & (thetas <= 0.15))

filtered_joint_llr = joint_llr[filtered_indices]
thetas = thetas[filtered_indices]
#print(max(thetas))
log_filtered_joint_llr=np.log(filtered_joint_llr)
#log_filtered_joint_llr=np.log(joint_llr)

bins = np.linspace(-0.15, 0.15,50) 
plt.hist(log_filtered_joint_llr, bins=bins, histtype='step', color='mediumblue', density=True,linewidth=2)
plt.hist(log_filtered_joint_llr, bins=bins, color='mediumblue', density=True, alpha=0.2)
print(np.where(log_filtered_joint_llr!=0.00000))
plt.plot(-0.5,0.5,label = "Signal + Backgrounds", linewidth = 2, color = "mediumblue")
plt.xlim(-0.1, 0.1)
#plt.yscale("log")
plt.xlabel(r'$\log r(x,z|\theta_0,\theta_1)$', size=14)
#plt.xlabel(r'$\hat{r}(x|\theta_0,\theta_1)$', size=14)
plt.ylabel('Normalized distribution', size=14)
plt.legend(frameon=False, fontsize=11)
plt.savefig("joint_llr_hist_cp_odd_alices.png", dpi=600,bbox_inches='tight')





################ALICES_joint_score########################################

#################
# thetas = np.load('/project/atlas/users/sabdadin/output/training_samples/alices_alices_gaussian_prior_0_1.2_10000_thetas_10e7_samples_CP_odd/theta0_train_ratio_ggf_combined_signal_0.npy') 
# joint_llr = np.load('/project/atlas/users/sabdadin/output/training_samples/alices_alices_gaussian_prior_0_1.2_10000_thetas_10e7_samples_CP_odd/t_xz_train_ratio_ggf_combined_signal_0.npy') 

# filtered_indices =  np.where((thetas >= -2.972) & (thetas <= 2.868))

# filtered_joint_llr = joint_llr[filtered_indices]

# # log_filtered_joint_llr=np.log(joint_llr)

# bins = np.linspace(-0.03, 0.03,30)  
# plt.hist(filtered_joint_llr, bins=bins, histtype='step', color='teal', density=True,linewidth=2)
# plt.hist(filtered_joint_llr, bins=bins, color='teal', density=True, alpha=0.2)
# plt.plot(-0.5,0.5,label = "Signal Only", linewidth = 2, color = "teal")


# thetas = np.load('/project/atlas/users/sabdadin/output/training_samples/alices_alices_gaussian_prior_0_1.2_10000_thetas_10e7_samples_CP_odd/theta0_train_ratio_ggf_signal_plus_ttbar_0.npy') 
# joint_llr = np.load('/project/atlas/users/sabdadin/output/training_samples/alices_alices_gaussian_prior_0_1.2_10000_thetas_10e7_samples_CP_odd/t_xz_train_ratio_ggf_signal_plus_ttbar_0.npy') 

# filtered_indices =  np.where((thetas >= -2.972) & (thetas <= 2.868))

# filtered_joint_llr = joint_llr[filtered_indices]
# thetas = thetas[filtered_indices]
# #print(max(thetas))
# # log_filtered_joint_llr=np.log(filtered_joint_llr)
# #log_filtered_joint_llr=np.log(joint_llr)
# bins = np.linspace(-0.03, 0.03, 50)  
# plt.hist(filtered_joint_llr, bins=bins, histtype='step', color='mediumblue', density=True,linewidth=2)
# plt.hist(filtered_joint_llr, bins=bins, color='mediumblue', density=True, alpha=0.2)
# print(np.where(filtered_joint_llr!=0.00000))
# plt.plot(-0.5,0.5,label = "Signal + Backgrounds", linewidth = 2, color = "mediumblue")
# plt.xlim(-0.05, 0.05)
# #plt.yscale("log")
# plt.xlabel(r'$t(x,z|\theta_0,\theta_1)$', size=14)
# #plt.xlabel(r'$\hat{r}(x|\theta_0,\theta_1)$', size=14)
# plt.ylabel('Normalized distribution', size=14)
# plt.legend(frameon=False, fontsize=11)
# plt.savefig("joint_score_hist_cp_odd_alices.pdf", dpi=600,bbox_inches='tight')


#########SALLY_score##########################
######################
# thetas = np.load('/project/atlas/users/sabdadin/output/training_samples/sally/theta_train_score_ggf_combined_signal_0.npy') 
# joint_llr = np.load('/project/atlas/users/sabdadin/output/training_samples/sally/t_xz_train_score_ggf_combined_signal_0.npy') 

# filtered_indices =  np.where((thetas >= -2.972) & (thetas <= 2.868))

# filtered_joint_llr = joint_llr[filtered_indices]

# # log_filtered_joint_llr=np.log(joint_llr)

# bins = np.linspace(-0.03, 0.03,30)  
# plt.hist(filtered_joint_llr, bins=bins, histtype='step', color='teal', density=True,linewidth=2)
# plt.hist(filtered_joint_llr, bins=bins, color='teal', density=True, alpha=0.2)
# plt.plot(-0.5,0.5,label = "Signal Only", linewidth = 2, color = "teal")


# thetas = np.load('/project/atlas/users/sabdadin/output/training_samples/sally/theta_train_score_ggf_signal_plus_ttbar_0.npy') 
# joint_llr = np.load('/project/atlas/users/sabdadin/output/training_samples/sally/t_xz_train_score_ggf_signal_plus_ttbar_0.npy') 

# filtered_indices =  np.where((thetas >= -2.972) & (thetas <= 2.868))

# filtered_joint_llr = joint_llr[filtered_indices]
# thetas = thetas[filtered_indices]
# #print(max(thetas))
# # log_filtered_joint_llr=np.log(filtered_joint_llr)
# #log_filtered_joint_llr=np.log(joint_llr)
# bins = np.linspace(-0.03, 0.03, 50)  
# plt.hist(filtered_joint_llr, bins=bins, histtype='step', color='mediumblue', density=True,linewidth=2)
# plt.hist(filtered_joint_llr, bins=bins, color='mediumblue', density=True, alpha=0.2)
# print(np.where(filtered_joint_llr!=0.00000))
# plt.plot(-0.5,0.5,label = "Signal + Backgrounds", linewidth = 2, color = "mediumblue")
# plt.xlim(-0.05, 0.05)
# #plt.yscale("log")
# plt.xlabel(r'$t(x,z|\theta_0,\theta_1)$', size=14)
# #plt.xlabel(r'$\hat{r}(x|\theta_0,\theta_1)$', size=14)
# plt.ylabel('Normalized distribution', size=14)
# plt.legend(frameon=False, fontsize=11)
# plt.savefig("joint_score_hist_cp_odd_sally.pdf", dpi=600,bbox_inches='tight')