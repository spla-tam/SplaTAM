import argparse
import os
import random
import sys
import shutil
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)

import cv2
import numpy as np
import torch
from tqdm import tqdm
import wandb

from datasets.gradslam_datasets import (load_dataset_config, ICLDataset, ReplicaDataset, ReplicaV2Dataset, AzureKinectDataset,
                                        ScannetDataset, Ai2thorDataset, Record3DDataset, RealsenseDataset, TUMDataset,
                                        ScannetPPDataset, NeRFCaptureDataset)
from utils.common_utils import seed_everything, save_params
from utils.recon_helpers import setup_camera
from utils.gs_helpers import (
    params2rendervar, params2depthplussilhouette,
    transformed_params2depthplussilhouette,
    transform_to_frame, report_progress, eval,
    l1_loss_v1, matrix_to_quaternion
)
from utils.gs_external import (
    calc_ssim, build_rotation, densify,
    get_expon_lr_func, update_learning_rate
)

from diff_gaussian_rasterization import GaussianRasterizer as Renderer


def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replicav2"]:
        return ReplicaV2Dataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["ai2thor"]:
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["record3d"]:
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["tum"]:
        return TUMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannetpp"]:
        return ScannetPPDataset(basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["nerfcapture"]:
        return NeRFCaptureDataset(basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True, 
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(), 
                                    torch.arange(height).cuda().float(),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((FX + FY)/2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method: {mean_sq_dist_method}")
    
    # Colorize point cloud
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
    point_cld = torch.cat((pts, cols), -1)

    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld


def initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution):
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {
        'means3D': means3D,
        'rgb_colors': init_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }

    # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))

    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float()}

    return params, variables


def initialize_optimizer(params, lrs_dict):
    lrs = lrs_dict
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]

    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def initialize_first_timestep(dataset, num_frames, lrs_dict, mean_sq_dist_method, gaussian_distribution):
    # Get RGB-D Data & Camera Parameters
    color, depth, intrinsics, pose = dataset[0]

    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    # Setup Camera
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())

    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    mask = (depth > 0) # Mask out invalid depth values
    mask = mask.reshape(-1)
    init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, intrinsics, w2c,
                                                mask=mask, compute_mean_sq_dist=True,
                                                mean_sq_dist_method=mean_sq_dist_method)

    # Initialize Parameters & Optimizer
    params, variables = initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution)
    optimizer = initialize_optimizer(params, lrs_dict)

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    variables['scene_radius'] = torch.max(depth)/2.0

    return params, variables, optimizer, intrinsics, w2c, cam


def get_loss_gs(params, curr_data, variables, loss_weights):
    # Initialize Loss Dictionary
    losses = {}

    # Initialize Render Variables
    rendervar = params2rendervar(params)
    depth_sil_rendervar = params2depthplussilhouette(params, curr_data['w2c'])

    # RGB Rendering
    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    # Depth & Silhouette Rendering
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    depth = depth_sil[0, :, :].unsqueeze(0)
    silhouette = depth_sil[1, :, :]

    # Get invalid Depth Mask
    valid_depth_mask = (curr_data['depth'] != 0.0)
    depth = depth * valid_depth_mask

    # RGB Loss
    losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
    
    # Depth Loss
    losses['depth'] = l1_loss_v1(depth, curr_data['depth'])

    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses


def initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))
    return params


def add_new_gaussians(params, variables, curr_data, sil_thres, 
                      time_idx, mean_sq_dist_method, gaussian_distribution):
    # Silhouette Rendering
    transformed_gaussians = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_gaussians)
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]
    non_presence_sil_mask = (silhouette < sil_thres)
    # Check for new foreground objects by using GT depth
    gt_depth = curr_data['depth'][0, :, :]
    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50*depth_error.median())
    # Determine non-presence mask
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)

    # Get the new frame Gaussians based on the Silhouette
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        valid_depth_mask = (curr_data['depth'][0, :, :] > 0)
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'], 
                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=mean_sq_dist_method)
        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution)
        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()

    return params, variables


def convert_params_to_store(params):
    params_to_store = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params_to_store[k] = v.detach().clone()
        else:
            params_to_store[k] = v
    return params_to_store


def offline_splatting(config: dict):
    # Print Config
    print("Loaded Config:")
    if "gaussian_distribution" not in config:
        config['gaussian_distribution'] = "anisotropic"
    print(f"{config}")

    # Init WandB
    if config['use_wandb']:
        wandb_step = 0
        wandb_time_step = 0
        wandb_run = wandb.init(project=config['wandb']['project'],
                               entity=config['wandb']['entity'],
                               group=config['wandb']['group'],
                               name=config['wandb']['name'],
                               config=config)
        wandb_run.define_metric("Mapping_Iters")
        wandb_run.define_metric("Number of Gaussians - Densification", step_metric="Mapping_Iters")
        wandb_run.define_metric("Learning Rate - Means3D", step_metric="Mapping_Iters")

    # Get Device
    device = torch.device(config["primary_device"])

    # Load Dataset
    print("Loading Dataset ...")
    dataset_config = config["data"]
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {}
        gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
    if "ignore_bad" not in dataset_config:
        dataset_config["ignore_bad"] = False
    if "use_train_split" not in dataset_config:
        dataset_config["use_train_split"] = True
    # Poses are relative to the first frame
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height_init"],
        desired_width=dataset_config["desired_image_width_init"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
    )

    mapping_dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
    )

    eval_dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["eval_stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
    )

    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)
    eval_num_frames = dataset_config["eval_num_frames"]
    if eval_num_frames == -1:
        eval_num_frames = len(eval_dataset)
    # Initialize Parameters, Optimizer & Canoncial Camera parameters
    params, variables, optimizer, intrinsics, w2c, cam = initialize_first_timestep(dataset, num_frames, 
                                                                                   config['train']['lrs_mapping'],
                                                                                   config['mean_sq_dist_method'],
                                                                                   config['gaussian_distribution'])

    _, _, map_intrinsics, _ = mapping_dataset[0]

    # Load all RGBD frames
    color_all_frames = []
    depth_all_frames = []
    gt_w2c_all_frames = []
    gs_cams_all_frames = []
    for time_idx in range(num_frames):
        color, depth, _, gt_pose = dataset[time_idx]
        # Process poses
        gt_w2c = torch.linalg.inv(gt_pose)
        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255
        depth = depth.permute(2, 0, 1)
        color_all_frames.append(color)
        depth_all_frames.append(depth)
        gt_w2c_all_frames.append(gt_w2c)
        # Setup Gaussian Splatting Camera
        gs_cam = setup_camera(color.shape[2], color.shape[1], 
                              intrinsics.cpu().numpy(), 
                              gt_w2c.detach().cpu().numpy())
        gs_cams_all_frames.append(gs_cam)

    # Load all RGBD frames - Mapping dataloader
    color_all_frames_map = []
    depth_all_frames_map = []
    gt_w2c_all_frames_map = []
    gs_cams_all_frames_map = []
    for time_idx in range(num_frames):
        color, depth, _, gt_pose = mapping_dataset[time_idx]
        # Process poses
        gt_w2c = torch.linalg.inv(gt_pose)
        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255
        depth = depth.permute(2, 0, 1)
        color_all_frames_map.append(color)
        depth_all_frames_map.append(depth)
        gt_w2c_all_frames_map.append(gt_w2c)
        # Setup Gaussian Splatting Camera
        gs_cam = setup_camera(color.shape[2], color.shape[1], 
                              map_intrinsics.cpu().numpy(), 
                              gt_w2c.detach().cpu().numpy())
        gs_cams_all_frames_map.append(gs_cam)

    # Iterate over Scan
    for time_idx in tqdm(range(num_frames)):
        # Optimization Iterations
        num_iters_mapping = config['train']['num_iters_mapping']

        # Initialize current frame data
        iter_time_idx = time_idx
        color = color_all_frames[iter_time_idx]
        depth = depth_all_frames[iter_time_idx]
        curr_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': iter_time_idx, 
                     'intrinsics': intrinsics, 'w2c': w2c, 'iter_gt_w2c_list': curr_gt_w2c}

        # Use GT Poses for Tracking
        with torch.no_grad():
            # Get the ground truth pose relative to frame 0
            rel_w2c = curr_gt_w2c[-1]
            rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
            rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
            rel_w2c_tran = rel_w2c[:3, 3].detach()
            # Update the camera parameters
            params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
            params['cam_trans'][..., time_idx] = rel_w2c_tran
        
        # Add new Gaussians to the scene based on the Silhouette
        if time_idx > 0:
            params, variables = add_new_gaussians(params, variables, curr_data, 
                                                  config['train']['sil_thres'], time_idx,
                                                  config['mean_sq_dist_method'], config['gaussian_distribution'])
        post_num_pts = params['means3D'].shape[0]
        if config['use_wandb']:
            wandb_run.log({"Init/Number of Gaussians": post_num_pts,
                           "Init/step": wandb_time_step})

        # Reset Optimizer & Learning Rates for Full Map Optimization
        optimizer = initialize_optimizer(params, config['train']['lrs_mapping'])
        means3D_scheduler = get_expon_lr_func(lr_init=config['train']['lrs_mapping']['means3D'], 
                                              lr_final=config['train']['lrs_mapping_means3D_final'],
                                              lr_delay_mult=config['train']['lr_delay_mult'],
                                              max_steps=config['train']['num_iters_mapping'])
        
        # Mapping
        if (time_idx + 1) == num_frames:
            if num_iters_mapping > 0:
                progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")
            for iter in range(num_iters_mapping):
                # Update Learning Rates for means3D
                updated_lr = update_learning_rate(optimizer, means3D_scheduler, iter+1)
                if config['use_wandb']:
                    wandb_run.log({"Learning Rate - Means3D": updated_lr})
                # Randomly select a frame until current time step
                iter_time_idx = random.randint(0, time_idx)
                # Initialize Data for selected frame
                iter_color = color_all_frames_map[iter_time_idx]
                iter_depth = depth_all_frames_map[iter_time_idx]
                iter_gt_w2c = gt_w2c_all_frames_map[:iter_time_idx+1]
                iter_gs_cam = gs_cams_all_frames_map[iter_time_idx]
                iter_data = {'cam': iter_gs_cam, 'im': iter_color, 'depth': iter_depth, 
                             'id': iter_time_idx, 'intrinsics': map_intrinsics, 
                             'w2c': gt_w2c_all_frames_map[iter_time_idx], 'iter_gt_w2c_list': iter_gt_w2c}
                # Loss for current frame
                loss, variables, losses = get_loss_gs(params, iter_data, variables, config['train']['loss_weights'])
                # Backprop
                loss.backward()
                with torch.no_grad():
                    # Gaussian-Splatting's Gradient-based Densification
                    if config['train']['use_gaussian_splatting_densification']:
                        params, variables = densify(params, variables, optimizer, iter, config['train']['densify_dict'])
                        if config['use_wandb']:
                            wandb_run.log({"Number of Gaussians - Densification": params['means3D'].shape[0]})
                    # Optimizer Update
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # Report Progress
                    if config['report_iter_progress']:
                        if config['use_wandb']:
                            report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['train']['sil_thres'], 
                                            wandb_run=wandb_run, wandb_step=wandb_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, online_time_idx=time_idx)
                        else:
                            report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['train']['sil_thres'], 
                                            mapping=True, online_time_idx=time_idx)
                    else:
                        progress_bar.update(1)
                    # Eval Params at 7K Iterations
                    if (iter + 1) == 7000:
                        print("Evaluating Params at 7K Iterations")
                        eval_params = convert_params_to_store(params)
                        output_dir = os.path.join(config["workdir"], config["run_name"])
                        eval_dir = os.path.join(output_dir, "eval_7k")
                        os.makedirs(eval_dir, exist_ok=True)
                        if config['use_wandb']:
                            eval(eval_dataset, eval_params, eval_num_frames, eval_dir, sil_thres=config['train']['sil_thres'],
                                 wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                                 mapping_iters=config["train"]["num_iters_mapping"], add_new_gaussians=True)
                        else:
                            eval(eval_dataset, eval_params, eval_num_frames, eval_dir, sil_thres=config['train']['sil_thres'],
                                 mapping_iters=config["train"]["num_iters_mapping"], add_new_gaussians=True)
            if num_iters_mapping > 0:
                progress_bar.close()

        # Increment WandB Step
        if config['use_wandb']:
            wandb_time_step += 1

    output_dir = os.path.join(config["workdir"], config["run_name"])
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # Evaluate Final Parameters
    with torch.no_grad():
        eval_params = convert_params_to_store(params)
        if config['use_wandb']:
            eval(eval_dataset, eval_params, eval_num_frames, eval_dir, sil_thres=config['train']['sil_thres'],
                 wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                 mapping_iters=config["train"]["num_iters_mapping"], add_new_gaussians=True)
        else:
            eval(eval_dataset, eval_params, eval_num_frames, eval_dir, sil_thres=config['train']['sil_thres'],
                 mapping_iters=config["train"]["num_iters_mapping"], add_new_gaussians=True)

    # Add Camera Parameters to Save them
    params = eval_params
    params['intrinsics'] = map_intrinsics.detach().cpu().numpy()
    params['w2c'] = w2c.detach().cpu().numpy()
    params['org_width'] = dataset_config["desired_image_width"]
    params['org_height'] = dataset_config["desired_image_height"]
    params['gt_w2c_all_frames'] = []
    for gt_w2c_tensor in gt_w2c_all_frames:
        params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
    params['gt_w2c_all_frames'] = np.stack(params['gt_w2c_all_frames'], axis=0)
    
    # Save Parameters
    save_params(params, output_dir)

    # Close WandB Run
    if config['use_wandb']:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])
    
    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    os.makedirs(results_dir, exist_ok=True)
    shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    offline_splatting(experiment.config)