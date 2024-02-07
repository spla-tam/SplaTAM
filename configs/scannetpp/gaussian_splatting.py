import os
from os.path import join as p_join

primary_device = "cuda:0"

scenes = ["8b5caf3398", "b20a261fdf"]

seed = 0

# Export SCENE env variable before running
os.environ["SCENE"] = "0"

# Train Split Eval
use_train_split = True

if use_train_split:
    scene_num_frames = [-1, 360]
else:
    scene_num_frames = [-1, -1]

scene_name = scenes[int(os.environ["SCENE"])]
num_frames = scene_num_frames[int(os.environ["SCENE"])]

full_res_width = 1168
full_res_height = 1752
downscale_factor = 2.0
densify_downscale_factor = 4.0

map_every = 1
keyframe_every = 5
mapping_window_size = 24
tracking_iters = 200
mapping_iters = 60

group_name = "ScanNet++_3DGS"
run_name = f"{scene_name}_{seed}"

config = dict(
    workdir=f"./experiments/{group_name}",
    run_name=run_name,
    seed=seed,
    primary_device=primary_device,
    map_every=map_every, # Mapping every nth frame
    keyframe_every=keyframe_every, # Keyframe every nth frame
    mapping_window_size=mapping_window_size, # Mapping window size
    report_global_progress_every=5, # Report Global Progress every nth frame
    eval_every=5, # Evaluate every nth frame (at end of SLAM)
    scene_radius_depth_ratio=3, # Max First Frame Depth to Scene Radius Ratio (For Pruning/Densification)
    mean_sq_dist_method="projective", # ["projective", "knn"] (Type of Mean Squared Distance Calculation for Scale of Gaussians)
    gaussian_distribution="isotropic", # ["isotropic", "anisotropic"] (Isotropic -> Spherical Covariance, Anisotropic -> Ellipsoidal Covariance)
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=False, # Save Checkpoints
    checkpoint_interval=5, # Checkpoint Interval
    use_wandb=True,
    wandb=dict(
        entity="theairlab",
        project="SplaTAM_CVPR_Rebuttal",
        group=group_name,
        name=run_name,
        save_qual=False,
        eval_save_qual=True,
    ),
    data=dict(
        dataset_name="scannetpp",
        basedir="/storage2/datasets/nkeetha/4d/data/ScanNetPP/data",
        sequence=scene_name,
        ignore_bad=False,
        use_train_split=True,
        desired_image_height=584,
        desired_image_width=876,
        desired_image_height_init=584,
        desired_image_width_init=876,
        start=0,
        end=-1,
        stride=1,
        num_frames=num_frames,
        eval_stride=1,
        eval_num_frames=-1,
    ),
    train=dict(
        num_iters_mapping=7000,
        sil_thres=0.5, # For Addition of new Gaussians & Visualization
        use_sil_for_loss=True, # Use Silhouette for Loss during Tracking
        loss_weights=dict(
            im=1.0,
            depth=0.0,
        ),
        lrs_mapping=dict(
            means3D=0.00032,
            rgb_colors=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.005,
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
        ),
        lrs_mapping_means3D_final=0.0000032,
        lr_delay_mult=0.01,
        use_gaussian_splatting_densification=True, # Use Gaussian Splatting-based Densification during Mapping
        densify_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=500,
            remove_big_after=3000,
            stop_after=15000,
            densify_every=100,
            grad_thresh=0.0002,
            num_to_split_into=2,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities=True,
            reset_opacities_every=3000, # Doesn't consider iter 0
        ),
    ),
    viz=dict(
        render_mode='color', # ['color', 'depth' or 'centers']
        offset_first_viz_cam=True, # Offsets the view camera back by 0.5 units along the view direction (For Final Recon Viz)
        show_sil=False, # Show Silhouette instead of RGB
        visualize_cams=True, # Visualize Camera Frustums and Trajectory
        viz_w=600, viz_h=340,
        viz_near=0.01, viz_far=100.0,
        view_scale=2,
        viz_fps=5, # FPS for Online Recon Viz
        enter_interactive_post_online=True, # Enter Interactive Mode after Online Recon Viz
    ),
)