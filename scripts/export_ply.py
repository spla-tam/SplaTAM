import os
import argparse
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

# Spherical harmonic constant
C0 = 0.28209479177387814


def rgb_to_spherical_harmonic(rgb):
    return (rgb-0.5) / C0


def spherical_harmonic_to_rgb(sh):
    return sh*C0 + 0.5


def save_ply(path, means, scales, rotations, rgbs, opacities, normals=None):
    path = Path(path)

    if normals is None:
        normals = np.zeros_like(means)

    colors = rgb_to_spherical_harmonic(rgbs)

    if scales.shape[1] == 1:
        scales = np.tile(scales, (1, 3))

    attrs = [
        'x',
        'y',
        'z',
        'nx',
        'ny',
        'nz',
        'f_dc_0',
        'f_dc_1',
        'f_dc_2',
        'opacity',
        'scale_0',
        'scale_1',
        'scale_2',
        'rot_0',
        'rot_1',
        'rot_2',
        'rot_3',
    ]

    dtype_full = [(attribute, 'f4') for attribute in attrs]
    elements = np.empty(means.shape[0], dtype=dtype_full)

    print("means shape: ", means.shape)
    print("normals shape: ", normals.shape)
    print("colors shape: ", colors.shape)
    print("opacities shape: ", opacities.shape)
    print("scales shape: ", scales.shape)
    print("rotations shape: ", rotations.shape)

    attributes = np.concatenate((means, normals, colors, opacities, scales, rotations), axis=1)
    print("attrs shape: ", attributes.shape)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load SplaTAM config
    experiment = SourceFileLoader(os.path.basename(args.config), args.config).load_module()
    config = experiment.config
    work_path = Path(config['workdir'])
    run_name = config['run_name']

    params = dict(np.load(work_path / run_name / "params.npz", allow_pickle=True))
    means = params['means3D']
    scales = params['log_scales']
    rotations = params['unnorm_rotations']
    rgbs = params['rgb_colors']
    opacities = params['logit_opacities']

    ply_path = work_path / "splat.ply"

    save_ply(ply_path, means, scales, rotations, rgbs, opacities)
