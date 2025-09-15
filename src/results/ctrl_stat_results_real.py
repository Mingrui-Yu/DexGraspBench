import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

parent_dir = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.append(parent_dir)

from util.grasp_controller import GraspController
from util.robot_adaptor import RobotAdaptor
from util.pin_helper import PinocchioHelper
from util.robots.base import RobotFactory, Robot, ArmHand
from mr_utils.utils_calc import skew, isometry3dToPosRotVec, sciR


# 提取文件名中的时间戳并排序（最新的排在最前面）
def extract_datetime(filename):
    # Remove extension if present
    name = os.path.splitext(filename)[0]
    # Take the last 5 parts: YYYY MM DD HH MM SS
    parts = name.split("_")
    timestamp_str = "_".join(parts[-2:])
    return datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")


def mean_without_outliers(data, k=2):
    data = np.asarray(data)
    mu = np.mean(data)
    sigma = np.std(data)
    mask = np.abs(data - mu) <= k * sigma
    return np.mean(data[mask])
    # return np.mean(data[mask]), mask  # 返回均值和掩码


if __name__ == "__main__":
    # robot: ArmHand = RobotFactory.create_robot(robot_type="ur5_leap_tac3d", arm_prefix="arm_", hand_prefix="rh_")
    robot: ArmHand = RobotFactory.create_robot(robot_type="dummy_arm_shadow", prefix="rh_")
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
    grasp_ctrl = GraspController(None, robot, robot_adaptor)

    # obj_lst = ["blue_box", "alcohol_bottle", "white_tape", "six_god", "banana", "mustard", "cheezit", "glass"]
    obj_lst = ["glass_i", "glass_t", "six_god_i", "six_god_t"]
    # obj_lst = ["glass"]
    method_lst = ["ours", "op", "bs1"]
    # method_lst = ["ours_t2", "ours_t4", "ours_t8", "ours_no_couple", "ours_mu1"]

    n_eval_step = 20
    n_eval_step_wrench = 5
    res = {
        "obj_pos_err_s2": [],
        "obj_rot_err_s2": [],
        "obj_rot_err_s3": [],
        "normalized_wrench": [],
    }

    for i_o, obj_name in enumerate(obj_lst):
        for key in res.keys():
            res[key].append([])

        for i_m, method in enumerate(method_lst):
            for key in res.keys():
                res[key][i_o].append([])

            data_dir = f"../adaptive_grasp_private/output/grasp_res/a_collect/{obj_name}/{method}"

            if not os.path.exists(data_dir):
                print(f"no file for {method}")
                continue
            files = [f for f in os.listdir(data_dir) if not f.endswith(".zip")]
            if not files:
                print(f"no file for {method}")
                continue

            if len(files) != 5:
                print(f"obj_name: {obj_name}, method: {method}")

            files_sorted = sorted(files, key=extract_datetime, reverse=False)

            for i_f, file in enumerate(files_sorted):
                data_path = os.path.join(data_dir, file, "data.npy")
                r_data = np.load(data_path, allow_pickle=True).item()

                n_step = len(r_data["dof"])
                seq_stage = np.asarray(r_data["stage"])

                # compute obj pose err
                # pos
                seq_obj_pose = np.asarray(r_data["obj_pose"])
                seq_obj_pos = seq_obj_pose[:, :3, 3]
                pos_err = seq_obj_pos - seq_obj_pos[0, :]
                pos_err = np.linalg.norm(pos_err, axis=-1)
                # rot
                seq_obj_rotmat = seq_obj_pose[:, :3, :3]
                rots = sciR.from_matrix(seq_obj_rotmat)
                R0 = rots[0]  # 注意要用 Rotation 对象
                R_rel = R0.inv() * rots
                angles = R_rel.magnitude()  # radians
                angles_deg = np.degrees(angles)

                pos_err_s2 = mean_without_outliers(pos_err[seq_stage == 2][-n_eval_step:])
                rot_err_s2 = mean_without_outliers(angles_deg[seq_stage == 2][-n_eval_step:])
                rot_err_s3 = mean_without_outliers(angles_deg[seq_stage == 3][-n_eval_step:])

                res["obj_pos_err_s2"][i_o][i_m].append(pos_err_s2)
                res["obj_rot_err_s2"][i_o][i_m].append(rot_err_s2)
                res["obj_rot_err_s3"][i_o][i_m].append(rot_err_s3)

                # compute normalized wrench
                seq_wrench = np.zeros((n_step, 6))
                seq_normalized_wrench = np.zeros((n_step, 6))

                for i in range(n_step):
                    contacts = r_data["contacts"][i]
                    n_con = len(contacts)
                    if len(contacts) > 0:
                        grasp_matrix = grasp_ctrl.compute_grasp_matrix(contacts)
                        cf_all = []
                        body_names = []
                        for contact in contacts:
                            body_names.append(contact["body1_name"])
                            cf = contact["contact_force"][:3].copy()
                            cf_all.append(cf)
                        cf_all = np.concatenate(cf_all, axis=0)

                        wrench = grasp_matrix @ cf_all.reshape(-1, 1)
                        seq_wrench[i, :] = wrench.reshape(-1)
                        normalized_wrench = grasp_ctrl.compute_normalized_wrench(grasp_matrix, cf_all)
                        seq_normalized_wrench[i, :] = normalized_wrench.reshape(-1)

                wrench_norm = np.linalg.norm(seq_normalized_wrench, axis=-1)
                normalized_wrench = np.mean(wrench_norm[-n_eval_step_wrench:])
                res["normalized_wrench"][i_o][i_m].append(normalized_wrench)

    for i_o, obj_name in enumerate(obj_lst):
        print(f"------------------ obj_name: {obj_name} -----------------")
        for key in res.keys():
            print(f"key: {key}")
            for i_m, method in enumerate(method_lst):
                print(f"method: {method}, mean: {np.mean(res[key][i_o][i_m])}, all: {res[key][i_o][i_m]}")

    print(" -----------------average across all objects --------------------------- ")
    for key in res.keys():
        data = np.asarray(res[key])
        print(f"key: {key}")
        for i_m, method in enumerate(method_lst):
            print(f"method: {method}, {np.mean(data[:, i_m])} +- {np.std(data[:, i_m])}")
