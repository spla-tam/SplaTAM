import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from natsort import natsorted

from .basedataset import GradSLAMDataset


def create_filepath_index_mapping(frames):
    return {frame["file_path"]: index for index, frame in enumerate(frames)}


class ScannetPPDataset(GradSLAMDataset):
    def __init__(
        self,
        basedir,
        sequence,
        ignore_bad: Optional[bool] = False,
        use_train_split: Optional[bool] = True,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 1168,
        desired_width: Optional[int] = 1752,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        config_dict = {}
        config_dict["dataset_name"] = "scannetpp"
        self.pose_path = None
        self.ignore_bad = ignore_bad
        self.use_train_split = use_train_split

        # Load Train & Test Split
        self.train_test_split = json.load(open(f"{self.input_folder}/dslr/train_test_lists.json", "r"))
        if self.use_train_split:
            self.image_names = self.train_test_split["train"]
        else:
            self.image_names = self.train_test_split["test"]
            self.train_image_names = self.train_test_split["train"]
        
        # Load NeRFStudio format camera & poses data
        self.cams_metadata = self.load_cams_metadata()
        if self.use_train_split:
            self.frames_metadata = self.cams_metadata["frames"]
            self.filepath_index_mapping = create_filepath_index_mapping(self.frames_metadata)
        else:
            self.frames_metadata = self.cams_metadata["test_frames"]
            self.train_frames_metadata = self.cams_metadata["frames"]
            self.filepath_index_mapping = create_filepath_index_mapping(self.frames_metadata)
            self.train_filepath_index_mapping = create_filepath_index_mapping(self.train_frames_metadata) 

        # Init Intrinsics
        config_dict["camera_params"] = {}
        config_dict["camera_params"]["png_depth_scale"] = 1000.0 # Depth is in mm
        config_dict["camera_params"]["image_height"] = self.cams_metadata["h"]
        config_dict["camera_params"]["image_width"] = self.cams_metadata["w"]
        config_dict["camera_params"]["fx"] = self.cams_metadata["fl_x"]
        config_dict["camera_params"]["fy"] = self.cams_metadata["fl_y"]
        config_dict["camera_params"]["cx"] = self.cams_metadata["cx"]
        config_dict["camera_params"]["cy"] = self.cams_metadata["cy"]

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

    def load_cams_metadata(self):
        cams_metadata_path = f"{self.input_folder}/dslr/nerfstudio/transforms_undistorted.json"
        cams_metadata = json.load(open(cams_metadata_path, "r"))
        return cams_metadata
    
    def get_filepaths(self):
        base_path = f"{self.input_folder}/dslr"
        color_paths = []
        depth_paths = []
        self.tmp_poses = []
        P = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ]
        ).float()
        if not self.use_train_split:
            self.first_train_image_name = self.train_image_names[0]
            self.first_train_image_index = self.train_filepath_index_mapping.get(self.first_train_image_name)
            self.first_train_frame_metadata = self.train_frames_metadata[self.first_train_image_index]
            # Get path of undistorted image and depth
            color_path = f"{base_path}/undistorted_images/{self.first_train_image_name}"
            depth_path = f"{base_path}/undistorted_depths/{self.first_train_image_name.replace('.JPG', '.png')}"
            color_paths.append(color_path)
            depth_paths.append(depth_path)
            # Get pose of first train frame in GradSLAM format
            c2w = torch.from_numpy(np.array(self.first_train_frame_metadata["transform_matrix"])).float()
            _pose = P @ c2w @ P.T
            self.tmp_poses.append(_pose)
        for image_name in self.image_names:
            # Search for image name in frames_metadata
            frame_metadata = self.frames_metadata[self.filepath_index_mapping.get(image_name)]
            # Check if frame is blurry and if it needs to be ignored
            if self.ignore_bad and frame_metadata['is_bad']:
                continue
            # Get path of undistorted image and depth
            color_path = f"{base_path}/undistorted_images/{image_name}"
            depth_path = f"{base_path}/undistorted_depths/{image_name.replace('.JPG', '.png')}"
            color_paths.append(color_path)
            depth_paths.append(depth_path)
            # Get pose of undistorted image in GradSLAM format
            c2w = torch.from_numpy(np.array(frame_metadata["transform_matrix"])).float()
            _pose = P @ c2w @ P.T
            self.tmp_poses.append(_pose)
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{base_path}/{self.embedding_dir}/*.pt"))
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        return self.tmp_poses

    def read_embedding_from_file(self, embedding_file_path):
        print(embedding_file_path)
        embedding = torch.load(embedding_file_path, map_location="cpu")
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)
