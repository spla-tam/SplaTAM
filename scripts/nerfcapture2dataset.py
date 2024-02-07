'''
Script to capture a dataset from the NeRFCapture iOS App. Code is adapted from instant-ngp/scripts/nerfcapture2nerf.py.
https://github.com/NVlabs/instant-ngp/blob/master/scripts/nerfcapture2nerf.py
'''
#!/usr/bin/env python3

import argparse
import os
import shutil
import sys
from pathlib import Path
import json
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

import cv2
import numpy as np

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types
from dataclasses import dataclass
from cyclonedds.domain import DomainParticipant, Domain
from cyclonedds.core import Qos, Policy
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from cyclonedds.util import duration


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/iphone/nerfcapture.py", type=str, help="Path to config file.")
    return parser.parse_args()


# DDS
# ==================================================================================================
@dataclass
@annotate.final
@annotate.autoid("sequential")
class SplatCaptureFrame(idl.IdlStruct, typename="SplatCaptureData.SplatCaptureFrame"):
    id: types.uint32
    annotate.key("id")
    timestamp: types.float64
    fl_x: types.float32
    fl_y: types.float32
    cx: types.float32
    cy: types.float32
    transform_matrix: types.array[types.float32, 16]
    width: types.uint32
    height: types.uint32
    image: types.sequence[types.uint8]
    has_depth: bool
    depth_width: types.uint32
    depth_height: types.uint32
    depth_scale: types.float32
    depth_image: types.sequence[types.uint8]


dds_config = """<?xml version="1.0" encoding="UTF-8" ?> \
<CycloneDDS xmlns="https://cdds.io/config" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="https://cdds.io/config https://raw.githubusercontent.com/eclipse-cyclonedds/cyclonedds/master/etc/cyclonedds.xsd"> \
    <Domain id="any"> \
        <Internal> \
            <MinimumSocketReceiveBufferSize>10MB</MinimumSocketReceiveBufferSize> \
        </Internal> \
        <Tracing> \
            <Verbosity>config</Verbosity> \
            <OutputFile>stdout</OutputFile> \
        </Tracing> \
    </Domain> \
</CycloneDDS> \
"""
# ==================================================================================================


def dataset_capture_loop(reader: DataReader, save_path: Path, overwrite: bool, n_frames: int, depth_scale: float):
    if save_path.exists():
        if overwrite:
            # Prompt user to confirm deletion
            if (input(f"warning! folder '{save_path}' will be deleted/replaced. continue? (Y/n)").lower().strip()+"y")[:1] != "y":
                sys.exit(1)
            shutil.rmtree(save_path)
        else:
            print(f"save_path {save_path} already exists")
            sys.exit(1)

    print("Waiting for frames...")
    # Make directory
    images_dir = save_path.joinpath("rgb")

    manifest = {
        "fl_x":  0.0,
        "fl_y":  0.0,
        "cx": 0.0,
        "cy": 0.0,
        "w": 0.0,
        "h": 0.0,
        "frames": []
    }

    total_frames = 0 # Total frames received

    # Start DDS Loop
    while True:
        sample = reader.read_next() # Get frame from NeRFCapture
        if sample:
            print(f"{total_frames + 1}/{n_frames} frames received")

            if total_frames == 0:
                save_path.mkdir(parents=True)
                images_dir.mkdir()
                manifest["w"] = sample.width
                manifest["h"] = sample.height
                manifest["cx"] = sample.cx
                manifest["cy"] = sample.cy
                manifest["fl_x"] = sample.fl_x
                manifest["fl_y"] = sample.fl_y
                manifest["integer_depth_scale"] = float(depth_scale)/65535.0
                if sample.has_depth:
                    depth_dir = save_path.joinpath("depth")
                    depth_dir.mkdir()

            # RGB
            image = np.asarray(sample.image, dtype=np.uint8).reshape((sample.height, sample.width, 3))
            cv2.imwrite(str(images_dir.joinpath(f"{total_frames}.png")), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            # Depth if avaiable
            depth = None
            if sample.has_depth:
                depth = np.asarray(sample.depth_image, dtype=np.uint8).view(
                    dtype=np.float32).reshape((sample.depth_height, sample.depth_width))
                depth = (depth*65535/float(depth_scale)).astype(np.uint16)
                depth = cv2.resize(depth, dsize=(
                    sample.width, sample.height), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(depth_dir.joinpath(f"{total_frames}.png")), depth)

            # Transform
            X_WV = np.asarray(sample.transform_matrix,
                              dtype=np.float32).reshape((4, 4)).T

            frame = {
                "transform_matrix": X_WV.tolist(),
                "file_path": f"rgb/{total_frames}.png",
                "fl_x": sample.fl_x,
                "fl_y": sample.fl_y,
                "cx": sample.cx,
                "cy": sample.cy,
                "w": sample.width,
                "h": sample.height
            }

            if depth is not None:
                frame["depth_path"] = f"depth/{total_frames}.png"

            manifest["frames"].append(frame)

            # Update index
            if total_frames == n_frames - 1:
                print("Saving manifest...")
                # Write manifest as json
                manifest_json = json.dumps(manifest, indent=4)
                with open(save_path.joinpath("transforms.json"), "w") as f:
                    f.write(manifest_json)
                print("Done")
                sys.exit(0)
            total_frames += 1


if __name__ == "__main__":
    args = parse_args()

    # Load config
    experiment = SourceFileLoader(
        os.path.basename(args.config), args.config
    ).load_module()

    # Setup DDS
    domain = Domain(domain_id=0, config=dds_config)
    participant = DomainParticipant()
    qos = Qos(Policy.Reliability.Reliable(
        max_blocking_time=duration(seconds=1)))
    topic = Topic(participant, "Frames", SplatCaptureFrame, qos=qos)
    reader = DataReader(participant, topic)

    config = experiment.config
    dataset_capture_loop(reader, Path(config['workdir']), config['overwrite'], config['num_frames'], config['depth_scale'])
