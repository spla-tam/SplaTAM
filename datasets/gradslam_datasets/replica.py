import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from natsort import natsorted

from .basedataset import GradSLAMDataset


class ReplicaDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "traj.txt")
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/results/frame*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/results/depth*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        for i in range(self.num_imgs):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)
    
class ReplicaV2Dataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        use_train_split: Optional[bool] = True,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.use_train_split = use_train_split
        if self.use_train_split:
            self.input_folder = os.path.join(basedir, sequence, "imap/00")
            self.pose_path = os.path.join(self.input_folder, "traj_w_c.txt")
        else:
            self.train_input_folder = os.path.join(basedir, sequence, "imap/00")
            self.train_pose_path = os.path.join(self.train_input_folder, "traj_w_c.txt")
            self.input_folder = os.path.join(basedir, sequence, "imap/01")
            self.pose_path = os.path.join(self.input_folder, "traj_w_c.txt")
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        if self.use_train_split:
            color_paths = natsorted(glob.glob(f"{self.input_folder}/rgb/rgb_*.png"))
            depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/depth_*.png"))
        else:
            first_train_color_path = f"{self.train_input_folder}/rgb/rgb_0.png"
            first_train_depth_path = f"{self.train_input_folder}/depth/depth_0.png"
            color_paths = [first_train_color_path] + natsorted(glob.glob(f"{self.input_folder}/rgb/rgb_*.png"))
            depth_paths = [first_train_depth_path] + natsorted(glob.glob(f"{self.input_folder}/depth/depth_*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []
        if not self.use_train_split:
            with open(self.train_pose_path, "r") as f:
                train_lines = f.readlines()
            first_train_frame_line = train_lines[0]
            first_train_frame_c2w = np.array(list(map(float, first_train_frame_line.split()))).reshape(4, 4)
            first_train_frame_c2w = torch.from_numpy(first_train_frame_c2w).float()
            poses.append(first_train_frame_c2w)
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        if self.use_train_split:
            num_poses = self.num_imgs
        else:
            num_poses = self.num_imgs - 1
        for i in range(num_poses):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)
    