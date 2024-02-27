from os.path import join as p_join

primary_device = "cuda:0"

base_dir = "./experiments/iPhone_Captures"
scene_name = "splatam_demo"
params_path = f"{base_dir}/{scene_name}/params.npz"

group_name = "iPhone_Captures"
run_name = f"{scene_name}_post_splatam_opt"

full_res_width = 1920
full_res_height = 1440
downscale_factor = 2.0
densify_downscale_factor = 4.0

config = dict(
    workdir=f"./experiments/{group_name}",
    run_name=run_name,
    seed=0,
    primary_device=primary_device,
    mean_sq_dist_method="projective", # ["projective", "knn"] (Type of Mean Squared Distance Calculation for Scale of Gaussians)
    gaussian_distribution="isotropic", # ["isotropic", "anisotropic"] (Isotropic -> Spherical Covariance, Anisotropic -> Ellipsoidal Covariance)
    report_iter_progress=False,
    use_wandb=False,
    wandb=dict(
        entity="theairlab",
        project="SplaTAM",
        group=group_name,
        name=run_name,
        save_qual=False,
        eval_save_qual=True,
    ),
    data=dict(
        dataset_name="nerfcapture",
        basedir=base_dir,
        sequence=scene_name,
        downscale_factor=downscale_factor,
        densify_downscale_factor=densify_downscale_factor,
        desired_image_height=int(full_res_height//downscale_factor),
        desired_image_width=int(full_res_width//downscale_factor),
        densification_image_height=int(full_res_height//densify_downscale_factor),
        densification_image_width=int(full_res_width//densify_downscale_factor),
        start=0,
        end=-1,
        stride=1,
        num_frames=-1,
        eval_stride=1,
        eval_num_frames=-1,
        param_ckpt_path=params_path,
    ),
    train=dict(
        num_iters_mapping=15000,
        sil_thres=0.5, # For Addition of new Gaussians & Visualization
        use_sil_for_loss=True, # Use Silhouette for Loss during Tracking
        loss_weights=dict(
            im=0.5,
            depth=1.0,
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