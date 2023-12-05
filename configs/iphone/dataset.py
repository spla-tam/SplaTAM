import os
from os.path import join as p_join

primary_device = "cuda:0"
seed = 0

base_dir = "./experiments/iPhone_Captures" # Root Directory to Save iPhone Dataset
scene_name = "dataset_demo" # Scan Name
num_frames = 10 # Desired number of frames to capture
depth_scale = 10.0 # Depth Scale used when saving depth
overwrite = False # Rewrite over dataset if it exists

config = dict(
    workdir=f"./{base_dir}/{scene_name}",
    overwrite=overwrite,
    depth_scale=depth_scale,
    num_frames=num_frames,
)