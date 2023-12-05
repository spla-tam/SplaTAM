import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from natsort import natsorted

from .basedataset import GradSLAMDataset


class AzureKinectDataset(GradSLAMDataset):
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
        self.pose_path = None

        # # check if a file named 'poses_global_dvo.txt' exists in the basedir / sequence folder
        # if os.path.isfile(os.path.join(basedir, sequence, "poses_global_dvo.txt")):
        #     self.pose_path = os.path.join(basedir, sequence, "poses_global_dvo.txt")

        if "odomfile" in kwargs.keys():
            self.pose_path = os.path.join(self.input_folder, kwargs["odomfile"])
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
        color_paths = natsorted(glob.glob(f"{self.input_folder}/color/*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        if self.pose_path is None:
            print("WARNING: Dataset does not contain poses. Returning identity transform.")
            return [torch.eye(4).float() for _ in range(self.num_imgs)]
        else:
            # Determine whether the posefile ends in ".log"
            # a .log file has the following format for each frame
            # frame_idx frame_idx+1
            # row 1 of 4x4 transform
            # row 2 of 4x4 transform
            # row 3 of 4x4 transform
            # row 4 of 4x4 transform
            # [repeat for all frames]
            #
            # on the other hand, the "poses_o3d.txt" or "poses_dvo.txt" files have the format
            # 16 entries of 4x4 transform
            # [repeat for all frames]
            if self.pose_path.endswith(".log"):
                # print("Loading poses from .log format")
                poses = []
                lines = None
                with open(self.pose_path, "r") as f:
                    lines = f.readlines()
                if len(lines) % 5 != 0:
                    raise ValueError(
                        "Incorrect file format for .log odom file " "Number of non-empty lines must be a multiple of 5"
                    )
                num_lines = len(lines) // 5
                for i in range(0, num_lines):
                    _curpose = []
                    _curpose.append(list(map(float, lines[5 * i + 1].split())))
                    _curpose.append(list(map(float, lines[5 * i + 2].split())))
                    _curpose.append(list(map(float, lines[5 * i + 3].split())))
                    _curpose.append(list(map(float, lines[5 * i + 4].split())))
                    _curpose = np.array(_curpose).reshape(4, 4)
                    poses.append(torch.from_numpy(_curpose))
            else:
                poses = []
                lines = None
                with open(self.pose_path, "r") as f:
                    lines = f.readlines()
                for line in lines:
                    if len(line.split()) == 0:
                        continue
                    c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
                    poses.append(torch.from_numpy(c2w))
            return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding  # .permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)
