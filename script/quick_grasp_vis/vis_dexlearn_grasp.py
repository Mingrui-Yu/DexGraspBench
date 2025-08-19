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

    object_path = "assets/object/DGN_2k"
    pc_path = "vision_data/azure_kinect_dk"
    object_pc_folder = os.path.join(object_path, pc_path)
    prefix = (
        "/home/mingrui/mingrui/research/adaptive_grasping_2/DexLearn/output/bodex_tabletop_shadow_nflow_debug0/tests/"
    )
    grasp_file_path = (
        "step_045000/core_bottle_1a7ba1f4c892e2da30711cdbdbc73924/tabletop_ur10e/scale008_pose002_0/partial_pc_02_0.npy"
    )
    grasp_data = np.load(os.path.join(prefix, grasp_file_path), allow_pickle=True).item()
    scene_path = grasp_data["scene_path"]
    scene_data = np.load(scene_path, allow_pickle=True).item()
    pc_path = os.path.join(object_pc_folder, scene_data["scene_id"], os.path.basename(grasp_file_path)).replace(
        "_0.npy", ".npy"
    )

    obj_name = scene_data["task"]["obj_name"]
    obj_pose = scene_data["scene"][obj_name]["pose"]
    obj_scale = scene_data["scene"][obj_name]["scale"]
    obj_mesh_path = scene_data["scene"][obj_name]["file_path"]
    obj_mesh_path = os.path.abspath(os.path.join(os.path.dirname(scene_path), obj_mesh_path))

    # pointcloud mesh
    pc = np.load(pc_path).reshape(-1, 3)
    colors = np.tile([0, 0, 255, 255], (pc.shape[0], 1))  # Blue in RGBA
    pc = tm.points.PointCloud(pc, colors=colors)

    # object mesh
    obj_transform = posQuat2Isometry3d(obj_pose[:3], quatWXYZ2XYZW(obj_pose[3:]))
    obj_mesh = tm.load_mesh(obj_mesh_path, process=False)
    obj_mesh = obj_mesh.copy().apply_scale(obj_scale)
    obj_mesh.apply_transform(obj_transform)

    grasp_qpos = grasp_data["grasp_qpos"]
    visualizer.set_robot_parameters(torch.tensor(grasp_qpos).unsqueeze(0))
    robot_mesh = visualizer.get_robot_trimesh_data(i=0, color=[255, 0, 0])

    axis = tm.creation.axis(origin_size=0.01, axis_length=1.0)
    scene = tm.Scene(geometry=[robot_mesh, obj_mesh, axis, pc])
    scene.show(smooth=False)

    # # grasp_pose_id = 0
    # for index in [3]:
    #     joint_names = grasp_info["joint_names"]
    #     robot_pose = grasp_info["grasp_poses"][0, [index], 0, :]

    #     print("index: ", index)
    #     print("robot_pose: ", robot_pose)

    #     visualize.set_robot_parameters(hand_pose=torch.tensor(robot_pose), joint_names=joint_names)
    #     robot_mesh_1 = visualize.get_robot_trimesh_data(i=0, color=[0, 255, 0])

    #     axis = tm.creation.axis(origin_size=0.01, axis_length=1.0)
    #     scene = tm.Scene(geometry=[robot_mesh_1, obj_mesh, table_mesh, axis])
    #     scene.show(smooth=False)
