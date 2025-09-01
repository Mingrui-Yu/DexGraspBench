import numpy as np

import torch
import trimesh as tm
import yaml
import os
import re
import json

import sys
from pathlib import Path

# Get parent of parent
parent_parent = Path(__file__).resolve().parents[2]
sys.path.append(str(parent_parent))

from mr_utils.utils_calc import posQuat2Isometry3d, quatWXYZ2XYZW
from src.util.robots.base import Robot, RobotFactory
from trimesh_visualizer import Visualizer


if __name__ == "__main__":
    hand = "leap_tac3d"
    robot = RobotFactory.create_robot(hand, prefix="rh_")
    robot_mjcf_path = robot.get_file_path("mjcf")
    pc_centering = True
    visualizer = Visualizer(robot_mjcf_path=robot_mjcf_path)

    grasp_file_path = "/home/mingrui/mingrui/research/adaptive_grasping_2/DexLearn/output/test/test.npy"
    grasp_data = np.load(grasp_file_path, allow_pickle=True).item()
    pc_path = "/home/mingrui/mingrui/research/adaptive_grasping_2/DexLearn/assets/object/DGN_2k/vision_data/azure_kinect_dk/core_bottle_1a7ba1f4c892e2da30711cdbdbc73924/tabletop_ur10e/scale006_pose000_0/partial_pc_00.npy"

    # pointcloud mesh
    pc = np.load(pc_path).reshape(-1, 3)

    # move to centroid
    if pc_centering:
        pc_centroid = np.mean(pc, axis=0, keepdims=True)
        pc = pc - pc_centroid
    colors = np.tile([0, 0, 255, 255], (pc.shape[0], 1))  # Blue in RGBA
    pc = tm.points.PointCloud(pc, colors=colors)

    for idx in range(10):
        for type in ["grasp_qpos"]:
            grasp_qpos = grasp_data[type][0, idx, ...]
            visualizer.set_robot_parameters(torch.tensor(grasp_qpos).unsqueeze(0))
            robot_mesh = visualizer.get_robot_trimesh_data(i=0, color=[255, 0, 0])

            axis = tm.creation.axis(origin_size=0.01, axis_length=1.0)
            scene = tm.Scene(geometry=[robot_mesh, axis, pc])
            scene.show(smooth=False)
