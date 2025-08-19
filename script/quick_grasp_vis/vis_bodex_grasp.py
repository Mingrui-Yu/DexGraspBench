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
    hand = "shadow"
    robot = RobotFactory.create_robot(hand, prefix="rh")
    robot_mjcf_path = robot.get_file_path("mjcf")
    mesh_dir_path = "/home/mingrui/mingrui/research/adaptive_grasping_2/DexGraspBench/third_party/mujoco_menagerie/shadow_hand/assets"

    visualizer = Visualizer(robot_mjcf_path=robot_mjcf_path)

    prefix = "/home/mingrui/mingrui/research/adaptive_grasping_2/BODex/src/curobo/content/assets/output/sim_shadow/tabletop/debug/graspdata/"
    grasp_file_path = "core_bottle_1a7ba1f4c892e2da30711cdbdbc73924/tabletop_ur10e/scale006_pose000_0_grasp.npy"
    grasp_data = np.load(os.path.join(prefix, grasp_file_path), allow_pickle=True).item()
    scene_path = str(grasp_data["scene_path"][0]).replace("adaptive_grasp", "adaptive_grasping_2")
    scene_data = np.load(scene_path, allow_pickle=True).item()

    joint_names = grasp_data["joint_names"]
    obj_name = scene_data["task"]["obj_name"]
    obj_pose = scene_data["scene"][obj_name]["pose"]
    obj_scale = scene_data["scene"][obj_name]["scale"]
    obj_mesh_path = scene_data["scene"][obj_name]["file_path"]
    obj_mesh_path = os.path.abspath(os.path.join(os.path.dirname(scene_path), obj_mesh_path))

    # obj_pc_path =

    # object mesh
    obj_transform = posQuat2Isometry3d(obj_pose[:3], quatWXYZ2XYZW(obj_pose[3:]))
    obj_mesh = tm.load_mesh(obj_mesh_path, process=False)
    obj_mesh = obj_mesh.copy().apply_scale(obj_scale)
    obj_mesh.apply_transform(obj_transform)

    for grasp_idx in range(20):
        grasp_qpos = grasp_data["robot_pose"][:, grasp_idx, 1, :]
        # TODO: an offset
        visualizer.set_robot_parameters(torch.tensor(grasp_qpos), joint_names=joint_names)
        robot_mesh = visualizer.get_robot_trimesh_data(i=0, color=[255, 0, 0])

        axis = tm.creation.axis(origin_size=0.01, axis_length=1.0)
        scene = tm.Scene(geometry=[robot_mesh, obj_mesh, axis])
        scene.show(smooth=False)
