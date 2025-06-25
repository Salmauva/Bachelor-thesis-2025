# import h5py
# import matplotlib.pyplot as plt
# import os

# filename = "/project/atlas/users/sabdadin/output/ggf_signal_plus_ttbar.h5"
# output_dir = "/project/atlas/users/sabdadin/plotscombine"

# os.makedirs(output_dir, exist_ok=True)
# with h5py.File(filename, "r") as f:
#     observables = f["samples/observations"][:]  # shape: (n_events, n_observables)
#     observable_names = [name.decode() for name in f["observables/names"][:]]

# # Plot first few observables as histograms
# for i, name in enumerate(observable_names[:len(observable_names)]):  # Change range as needed
#     plt.figure()
#     plt.hist(observables[:, i], bins=50, histtype="step", color="navy")
#     plt.title(name)
#     plt.xlabel(name)
#     plt.ylabel("Counts")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, f"observable_{i}_{name}.png"))
#     plt.close()


# # # Example of how to read the HDF5 file structure and see the contents   
# # import h5py

# # filename = "/project/atlas/users/sabdadin/output/ggf_signal_plus_ttbar.h5"

# # with h5py.File(filename, "r") as f:
# #     def print_structure(name, obj):
# #         print(name)
# #     f.visititems(print_structure)
######################################asPDF######################################
import h5py
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# Define the file paths
filename = "/project/atlas/users/sabdadin/output/ggf_signal_plus_ttbar.h5"
output_pdf = "/project/atlas/users/sabdadin/plotscombine/combined_observables.pdf"

# Create the output directory if it doesn't exist
output_dir = "/project/atlas/users/sabdadin/plotscombine"
os.makedirs(output_dir, exist_ok=True)

# Open the HDF5 file and read the observables
with h5py.File(filename, "r") as f:
    observables = f["samples/observations"][:]  # shape: (n_events, n_observables)
    observable_names = [name.decode() for name in f["observables/names"][:]]

# Create a PDF document to save all the figures
with PdfPages(output_pdf) as pdf:
    for i, name in enumerate(observable_names[:len(observable_names)]):  # Change range as needed
        plt.figure()
        plt.hist(observables[:, i], bins=50, histtype="step", color="navy")
        plt.title(name)
        plt.xlabel(name)
        plt.ylabel("Counts")
        plt.grid(True)
        plt.tight_layout()
        
        # Save the current figure to the PDF
        pdf.savefig()  # Saves the current figure to the pdf
        plt.close()

print(f"All figures have been saved to {output_pdf}")
#making individual histograms for SM, CP odd, CP even, and background for each observable and plotting in one graph per onbservable saving it to same pdf as above
filename_new_plot_SM = "/data/atlas/users/mfernand/bachelor_project_samples/ggF_smeftsim_SM.h5"
filename_new_plot_neg = "/data/atlas/users/mfernand/bachelor_project_samples/ggF_smeftsim_neg_chgtil.h5"
filename_new_plot_pos = "/data/atlas/users/mfernand/bachelor_project_samples/ggF_smeftsim_pos_chgtil.h5" 
filename_new_plot_ttbar = "/data/atlas/users/mfernand/bachelor_project_samples/ttbar.h5"

# Open the HDF5 file and read the observables
with h5py.File(filename_new_plot_SM, "r") as f:
    observables_SM = f["samples/observations"][:]  # shape: (n_events, n_observables)
# Open the HDF5 file and read the observables
with h5py.File(filename_new_plot_neg, "r") as f:
    observables_neg = f["samples/observations"][:]  # shape: (n_events, n_observables)
with h5py.File(filename_new_plot_pos, "r") as f:
    observables_pos = f["samples/observations"][:]  # shape: (n_events, n_observables)
with h5py.File(filename_new_plot_ttbar, "r") as f:
    observables_ttbar = f["samples/observations"][:]  # shape: (n_events, n_observables)

# Create a PDF document to save all the figures
with PdfPages(output_pdf) as pdf:
    for i, name in enumerate(observable_names[:len(observable_names)]):  # Change range as needed
        plt.figure()
        plt.hist(observables[:, i], bins=50, histtype="step", color="black")
        plt.hist(observables_SM[:, i], bins=50, histtype="step", color="navy")
        plt.hist(observables_neg[:, i], bins=50, histtype="step", color="red")
        plt.hist(observables_pos[:, i], bins=50, histtype="step", color="green")
        plt.hist(observables_ttbar[:, i], bins=50, histtype="step", color="orange")
        plt.legend(['All signals',"SM", "Neg_chgtil=-2.972", "Pos_chgtil=2.868", "ttbar"])
        plt.title(name)
        plt.xlabel(name)
        plt.ylabel("Counts")
        plt.grid(True)
        plt.tight_layout()
        
        # Save the current figure to the PDF
        pdf.savefig()  # Saves the current figure to the pdf
        plt.close()
