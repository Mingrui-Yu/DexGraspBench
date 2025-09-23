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
    setting_name = "dist_2"
    hand = "leap_tac3d"
    method_lst = ["ours_ab2"]
    all_res = {}

    for method in method_lst:
        all_res[method] = task_control_stat(setting_name, hand, method)

    metrics = ["obj_pos", "obj_rot", "wrench"]
    ours = all_res["ours_ab2"]
    other_methods = [m for m in all_res.keys() if m != "ours_ab2"]
    top_n = 10  # 想看前10个显著差异样本
    thres_dct = {
        "obj_pos": 0.002,
        "obj_rot": 2,
        "wrench": 0.2,
    }

    # 存储每个指标的显著性排序
    mask_all = np.ones((len(ours["obj_pos"])), dtype=bool)
    for metric in metrics:
        ours_metric = ours[metric]
        thres = thres_dct[metric]
        mask = ours_metric < thres
        mask_all = mask_all & mask

    valid_indices = np.where(mask_all)[0]

    # 先把 case_id -> pos_id 列表存起来
    case_dict = {}
    for idx in valid_indices:
        if setting_name == "dist_0":
            # dist_0 只有 case_id，不需要 pos_id
            case_id = idx
            pos_id = None
        elif setting_name == "dist_2":
            case_id = idx // 8
            pos_id = idx % 8
        else:
            raise ValueError(f"Unknown setting_name: {setting_name}")

        if case_id not in case_dict:
            case_dict[case_id] = []
        if pos_id is not None:
            case_dict[case_id].append(pos_id)

    # 输出
    for case_id, pos_list in case_dict.items():
        if setting_name == "dist_0":
            print(f"case_id {case_id}:")
        elif setting_name == "dist_2":
            pos_str = ", ".join(str(p) for p in pos_list)
            print(f"case_id {case_id}, pos_id(s) {pos_str}:")

        # for m in ["ours"]:
        #     line = [f"{m}"]
        #     for met in metrics:
        #         # 取其中一个 idx（多个 pos_id 时选第一个）
        #         idx = case_id if setting_name == "dist_0" else case_id * 8 + pos_list[0]
        #         val = all_res[m][met][idx] if m != "ours" else ours[met][idx]
        #         if np.ndim(val) > 0:
        #             val_str = "[" + ", ".join(f"{v:.4f}" for v in np.ravel(val)) + "]"
        #         else:
        #             val_str = f"{val:.4f}"
        #         line.append(f"{met}={val_str}")
        #     print(" | ".join(line))
        # print()

    # for idx in valid_indices:
    #     if setting_name == "dist_0":
    #         print(f"case_id: {idx}")
    #     elif setting_name == "dist_2":
    #         case_id = idx // 8
    #         pos_id = idx % 8
    #         print(f"case_id {case_id}, pos_id {pos_id}:")

    # for m in ["ours"]:
    #     line = [f"{m}"]
    #     for met in metrics:
    #         val = all_res[m][met][idx] if m != "ours" else ours[met][idx]
    #         # 如果是向量，就逐元素打印
    #         if np.ndim(val) > 0:
    #             val_str = "[" + ", ".join(f"{v:.4f}" for v in np.ravel(val)) + "]"
    #         else:
    #             val_str = f"{val:.4f}"
    #         line.append(f"{met}={val_str}")
    #     print(" | ".join(line))
    # print()

    print(list(valid_indices))


if __name__ == "__main__":
    main()
