from madminer.sampling import combine_and_shuffle
import logging
import os

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)

# Path to your samples
base_dir = "/data/atlas/users/mfernand/bachelor_project_samples"
output_dir = "/project/atlas/users/sabdadin/output"

# Files to combine
ggf_sm = os.path.join(base_dir, "ggF_smeftsim_SM.h5")
ggf_neg = os.path.join(base_dir, "ggF_smeftsim_neg_chgtil.h5")
ggf_pos = os.path.join(base_dir, "ggF_smeftsim_pos_chgtil.h5")
ttbar = os.path.join(base_dir, "ttbar.h5")

# 1. Combine all ggF signal (SM + BSM) together
combine_and_shuffle(
    [ggf_sm, ggf_neg, ggf_pos],
    os.path.join(output_dir, "ggf_combined_signal.h5")
)

# 2. Combine all ggF signal + ttbar background
combine_and_shuffle(
    [ggf_sm, ggf_neg, ggf_pos, ttbar],
    os.path.join(output_dir, "ggf_signal_plus_ttbar.h5")
)

# 3. Optionally: Compare each individually to ttbar
combine_and_shuffle(
    [ggf_sm, ttbar],
    os.path.join(output_dir, "ggf_sm_plus_ttbar.h5")
)

combine_and_shuffle(
    [ggf_neg, ttbar],
    os.path.join(output_dir, "ggf_neg_plus_ttbar.h5")
)

combine_and_shuffle(
    [ggf_pos, ttbar],
    os.path.join(output_dir, "ggf_pos_plus_ttbar.h5")
)
import shutil

# Define the destination directory
destination_dir = "/project/atlas/users/sabdadin/output"

# Copy each file to the destination directory
shutil.copy(ggf_sm, destination_dir)
shutil.copy(ggf_neg, destination_dir)
shutil.copy(ggf_pos, destination_dir)
shutil.copy(ttbar, destination_dir)