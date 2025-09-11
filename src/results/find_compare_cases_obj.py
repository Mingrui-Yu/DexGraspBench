import numpy as np
import os
import sys
from glob import glob
import multiprocessing
import logging
from pathlib import Path
from tqdm import tqdm
import yaml


parent_dir = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.append(parent_dir)

from util.rot_util import torch_quaternion_to_matrix, torch_matrix_to_axis_angle
from util.grasp_controller import GraspController
from util.robot_adaptor import RobotAdaptor
from util.pin_helper import PinocchioHelper
from util.robots.base import RobotFactory, Robot, ArmHand

from mr_utils.utils_torch import quaternion_angular_error
from mr_utils.utils_calc import quatWXYZ2XYZW, sciR


def read_data(npy_path):
    data = np.load(npy_path, allow_pickle=True).item()
    return data


def read_data_with_index(args):
    idx, npy_path = args
    return idx, read_data(npy_path)


def get_control_results(data_lst, hand_name):
    robot_prefix = "rh_" if "allegro" not in hand_name else ""
    robot: ArmHand = RobotFactory.create_robot(robot_type=hand_name, prefix=robot_prefix)

    robot_file_path = robot.get_file_path("mjcf")
    dof_names = robot.dof_names
    doa_names = robot.doa_names
    doa2dof_matrix = robot.doa2dof_matrix

    robot_model = PinocchioHelper(robot_file_path=robot_file_path, robot_file_type="mjcf")
    robot_adaptor = RobotAdaptor(
        robot_model=robot_model,
        dof_names=dof_names,
        doa_names=doa_names,
        doa2dof_matrix=doa2dof_matrix,
    )
    grasp_ctrl = GraspController(configs=None, robot=robot, robot_adaptor=robot_adaptor)

    lift_height = 0.2
    n_s = 5

    err_obj_pos_all = []
    err_obj_angle_all = []
    sum_cf_all = []
    wrench_all = []
    normalized_wrench_all = []
    success_cases = []
    failure_cases = []
    invalid_cases = []

    for i, r_data in enumerate(tqdm(data_lst, desc="Computing results")):
        if r_data["obj_pose"] == []:
            invalid_cases.append(i)

            err_obj_pos_all.append(1e2)
            err_obj_angle_all.append(1e2)
            normalized_wrench_all.append(1e2)
            continue

        seq_obj_pose = np.asarray(r_data["obj_pose"])
        # obj poses
        init_obj_pose = seq_obj_pose[0, :]
        target_obj_pose = init_obj_pose.copy()
        target_obj_pose[2] += lift_height
        final_obj_pose = seq_obj_pose[-1, :]
        err_obj_pos = np.linalg.norm(target_obj_pose[:3] - final_obj_pose[:3])
        err_obj_pos_z = np.linalg.norm(target_obj_pose[2] - final_obj_pose[2])
        err_obj_angle = (
            sciR.from_quat(quatWXYZ2XYZW(target_obj_pose[3:])).inv() * sciR.from_quat(quatWXYZ2XYZW(final_obj_pose[3:]))
        ).magnitude()
        err_obj_angle = np.rad2deg(err_obj_angle)

        # contact forces
        T = len(r_data["contacts"])
        seq_wrench = np.zeros((T, 6))
        seq_normalized_wrench = np.zeros((T, 6))
        seq_sum_cf = np.zeros((T))
        for t in range(T):
            contacts = r_data["contacts"][t]
            n_con = len(contacts)
            if n_con > 0:
                grasp_matrix = grasp_ctrl.compute_grasp_matrix(contacts)
                cf_all = np.concatenate([c["contact_force"][:3] for c in contacts], axis=0)
                wrench = grasp_matrix @ cf_all.reshape(-1, 1)
                normalized_wrench = grasp_ctrl.compute_normalized_wrench(grasp_matrix, cf_all)
                sum_cf = np.sum(cf_all.reshape(-1, 3)[:, 0])
                seq_wrench[t, :] = wrench.reshape(-1)
                seq_normalized_wrench[t, :] = normalized_wrench.reshape(-1)
                seq_sum_cf[t] = sum_cf

        if err_obj_pos_z < lift_height / 2.0:  # regard as successful grasp
            success_cases.append(i)
        else:
            failure_cases.append(i)

        err_obj_pos_all.append(err_obj_pos)
        err_obj_angle_all.append(err_obj_angle)
        normalized_wrench_all.append(np.mean(np.linalg.norm(seq_normalized_wrench[-n_s:], axis=-1)))

    err_obj_pos_all = np.asarray(err_obj_pos_all)
    err_obj_angle_all = np.asarray(err_obj_angle_all)
    normalized_wrench_all = np.asarray(normalized_wrench_all)

    res = {}
    res["obj_pos"] = err_obj_pos_all
    res["obj_rot"] = err_obj_angle_all
    res["wrench"] = normalized_wrench_all

    return res

    # index = success_cases[np.argmax(err_obj_angle_all)]
    # grasp_id = index // 8
    # pos_id = index % 8
    # print(f"max obj rot err: {np.max(err_obj_angle_all)}, grasp_id: {grasp_id}, pos_id: {pos_id}")


def task_control_stat(setting_name, hand, method):
    n_worker = 12

    control_dir = f"output/learn_dummy_arm_{hand}/control"
    control_lst = glob(os.path.join(control_dir, "**/*.npy"), recursive=True)

    control_lst = [p for p in control_lst if Path(p).match(f"*/{method}/*.npy") and setting_name in p]
    # control_lst = [x for x in control_lst if method in x and "pos_0" not in x]
    control_lst = sorted(control_lst)
    logging.info(f"Find {len(control_lst)} grasp data using control method '{method}' in {control_dir}.")

    with multiprocessing.Pool(processes=n_worker) as pool:
        result_iter = pool.imap_unordered(read_data_with_index, enumerate(control_lst))
        data_lst = [None] * len(control_lst)
        for idx, data in result_iter:  # keep the original order
            data_lst[idx] = data

    res = get_control_results(data_lst, hand_name=hand)
    return res


def main():
    setting_name = "dist_0"
    hand = "shadow"
    method_lst = ["ours_ab2", "op", "bs1", "bs2", "bs3"]

    all_res = {}

    for method in method_lst:
        all_res[method] = task_control_stat(setting_name, hand, method)

    metrics = ["obj_pos", "obj_rot", "wrench"]
    ours = all_res["ours_ab2"]
    other_methods = [m for m in all_res.keys() if m != "ours_ab2"]

    top_n = 10  # 想看前10个显著差异样本

    # 存储每个指标的显著性排序
    significant_cases = {}

    for metric in metrics:
        ours_metric = ours[metric]
        num_cases = len(ours_metric)

        # 累加其他方法的差异
        diff_scores = np.zeros(num_cases)
        for m in other_methods:
            diff = np.abs(ours_metric - all_res[m][metric])
            # 如果 metric 是多维向量，比如 obj_pos 是 3D，可以先取 norm
            if diff.ndim > 1:
                diff = np.linalg.norm(diff, axis=-1)
            diff_scores += diff  # 可以累加或取最大 np.maximum(diff_scores, diff)

        # 按差异从大到小排序
        sorted_idx = np.argsort(-diff_scores)
        significant_cases[metric] = {"indices": sorted_idx[:top_n], "scores": diff_scores[sorted_idx[:top_n]]}

    # 输出结果
    for metric, info in significant_cases.items():
        print(f"Top {top_n} significant cases for {metric}:")
        for idx, score in zip(info["indices"], info["scores"]):
            print(f"  case {idx}: score={score}")


if __name__ == "__main__":
    main()
