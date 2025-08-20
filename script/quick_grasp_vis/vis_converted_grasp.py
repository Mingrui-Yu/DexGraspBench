import numpy as np

import torch
import trimesh as tm
import yaml
import os
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
    hand = "allegro"
    robot = RobotFactory.create_robot(hand, prefix="rh")
    robot_mjcf_path = robot.get_file_path("mjcf")

    visualizer = Visualizer(robot_mjcf_path=robot_mjcf_path)

    object_path = "assets/object/DGN_2k"
    prefix = "/home/mingrui/mingrui/research/adaptive_grasping_2/DexGraspBench/output/bodex_allegro/graspdata/"
    grasp_file_path = "mujoco_Seagate_Archive_HDD_8_TB_Internal_hard_drive_SATA_6Gbs_35_ST8000AS0002/tabletop_ur10e/scale012_pose003_0/0_grasp.npy"
    grasp_data = np.load(os.path.join(prefix, grasp_file_path), allow_pickle=True).item()
    scene_path = grasp_data["scene_path"]
    scene_data = np.load(scene_path, allow_pickle=True).item()

    obj_name = scene_data["task"]["obj_name"]
    obj_pose = scene_data["scene"][obj_name]["pose"]
    obj_scale = scene_data["scene"][obj_name]["scale"]
    obj_mesh_path = scene_data["scene"][obj_name]["file_path"]
    obj_mesh_path = os.path.abspath(os.path.join(os.path.dirname(scene_path), obj_mesh_path))

    # object mesh
    obj_transform = posQuat2Isometry3d(obj_pose[:3], quatWXYZ2XYZW(obj_pose[3:]))
    obj_mesh = tm.load_mesh(obj_mesh_path, process=False)
    obj_mesh = obj_mesh.copy().apply_scale(obj_scale)
    obj_mesh.apply_transform(obj_transform)

    grasp_qpos = grasp_data["grasp_qpos"]
    visualizer.set_robot_parameters(torch.tensor(grasp_qpos).unsqueeze(0))
    robot_mesh = visualizer.get_robot_trimesh_data(i=0, color=[255, 0, 0])

    axis = tm.creation.axis(origin_size=0.01, axis_length=1.0)
    scene = tm.Scene(geometry=[robot_mesh, obj_mesh, axis])
    scene.show(smooth=False)
