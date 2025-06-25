# file to read the number of samples in h5 file after morphing (so #events will be reduced)
import h5py

def explore_group(group, prefix=""):
    for key in group.keys():
        item = group[key]
        full_key = f"{prefix}/{key}" if prefix else key
        if isinstance(item, h5py.Dataset):
            print(f"{full_key}: {item.shape}")
        elif isinstance(item, h5py.Group):
            print(f"{full_key}/ (group)")
            explore_group(item, full_key)

with h5py.File("/project/atlas/users/sabdadin/output/ggf_combined_signal.h5", "r") as f:
    explore_group(f)
