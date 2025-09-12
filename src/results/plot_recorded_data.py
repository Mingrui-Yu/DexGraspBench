import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
from omegaconf import OmegaConf

parent_dir = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.append(parent_dir)

from util.grasp_controller import GraspController
from util.robot_adaptor import RobotAdaptor
from util.pin_helper import PinocchioHelper
from util.robots.base import RobotFactory, Robot, ArmHand
from mr_utils.utils_calc import skew

if __name__ == "__main__":
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

    config = OmegaConf.load("config/task/control_eval.yaml").control
    grasp_ctrl = GraspController(config, robot, robot_adaptor)

    method = "ours"
    data_path = "output/learn_dummy_arm_shadow/control/core_jar_32dc55c3e945384dbc5e533ab711fd24/tabletop_ur10e/scale010_pose008_0/ours_ab2/partial_pc_02_2_dist_2_pos_6.npy"

    r_data = np.load(data_path, allow_pickle=True).item()

    n_step = len(r_data["dof"])

    seq_dof = np.asarray(r_data["dof"])
    seq_doa = np.asarray(r_data["doa"])
    seq_planned_dof = np.asarray(r_data["planned_dof"])

    dt = 0.1
    t = np.arange(n_step) * dt

    # --------------------------------
    plt.figure(figsize=(8, 5))
    plt.title("sum of normal contact force")

    seq_cf = []
    for i in range(n_step):
        contacts = r_data["contacts"][i]
        sum_cf = 0
        for contact in contacts:
            cf = contact["contact_force"]
            cf_mag = cf[0]  # normal force
            sum_cf += cf_mag
        seq_cf.append(sum_cf)
    seq_cf = np.asarray(seq_cf)

    plt.plot(t, seq_cf)

    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    # plt.ylim([0, 10])

    # --------------------------------
    plt.figure(figsize=(8, 5))
    plt.title("arm dof")

    labels = ["x", "y", "z", "rx", "ry", "rz"]
    delta_dof = seq_dof[:, 0:6] - seq_dof[0, 0:6]
    delta_planned_dof = seq_planned_dof[:, 0:6] - seq_planned_dof[0, 0:6]
    plt.plot(t, delta_dof, label=labels)
    plt.plot(t, delta_planned_dof, label=[f"planned_{n}" for n in labels])

    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.ylim([-0.02, 0.02])

    # --------------------------------
    plt.figure(figsize=(8, 5))
    plt.title("hand dof")

    labels = [i for i in range(robot.hand.n_dof)]
    delta_dof = seq_dof[:, -robot.hand.n_dof :] - seq_dof[0, -robot.hand.n_dof :]
    delta_planned_dof = seq_planned_dof[:, -robot.hand.n_dof :] - seq_planned_dof[0, -robot.hand.n_dof :]
    plt.plot(t, delta_dof, label=labels)
    plt.plot(t, delta_planned_dof, label=[f"planned_{n}" for n in labels])

    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    # plt.ylim([-0.02, 0.02])

    # --------------------------------
    plt.figure(figsize=(8, 5))
    plt.title("hand doa")

    labels = [i for i in range(robot.hand.n_dof)]
    delta_doa = seq_doa[:, -robot.hand.n_dof :] - seq_doa[0, -robot.hand.n_dof :]
    delta_dof = seq_dof[:, -robot.hand.n_dof :] - seq_dof[0, -robot.hand.n_dof :]
    plt.plot(t, delta_doa, label=[f"doa_{n}" for n in labels])
    plt.plot(t, delta_dof, label=[f"dof_{n}" for n in labels])

    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    # --------------------------------
    plt.figure(figsize=(8, 5))
    plt.title("contact model")

    seq_cf_actual = np.zeros((t.shape[0], 3))
    seq_cf_pred = np.zeros((t.shape[0], 3))
    seq_delta_p_1 = np.zeros((t.shape[0], 3))
    seq_delta_p_2 = np.zeros((t.shape[0], 3))

    n_arm_dof = 6
    I3 = np.eye(3)
    body_name = "rh_thdistal"
    # body_name = "rh_ffdistal"

    for i in range(n_step):
        contacts = r_data["contacts"][i]
        robot_adaptor.compute_jaco_a(seq_doa[i, :])
        delta_q = seq_doa[i, :] - seq_dof[i, :]

        if len(contacts) > 0:
            _, stacked = grasp_ctrl.Ks(q_a=seq_doa[i, :], q_f=seq_dof[i, :], contacts=contacts)
            contact_force_all = np.concatenate([contact["contact_force"][:3] for contact in contacts], axis=0)

            contact_jaco = stacked["jaco_a"]
            contact_jaco[:, :n_arm_dof] = 0
            Ks = stacked["Ks_h"]
            delta_p_1 = contact_jaco @ delta_q.reshape(-1, 1)
            cf_pred = (Ks @ delta_p_1).reshape(-1)

            contact_body_lst = [contact["body1_name"] for contact in contacts]
            if body_name in contact_body_lst:
                contact_idx = [contact["body1_name"] for contact in contacts].index(body_name)
                seq_cf_actual[i, :] = contact_force_all.reshape(-1, 3)[contact_idx, :]
                seq_cf_pred[i, :] = cf_pred.reshape(-1, 3)[contact_idx, :]
                seq_delta_p_1[i, :] = delta_p_1.reshape(-1, 3)[contact_idx, :]

    labels = ["x", "y", "z"]
    for i, n in enumerate(labels):
        # 画实线 (actual)
        plt.plot(t, seq_cf_actual[:, i], label=n)
        # 画虚线 (pred)，颜色和前一条线一致
        plt.plot(t, seq_cf_pred[:, i], linestyle="--", color=plt.gca().lines[-1].get_color(), label=f"pred_{n}")

    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    # # --------------------------------
    # plt.figure(figsize=(8, 5))
    # plt.title("delta_p v.s. J delta_q")

    # plt.plot(t, seq_delta_p_1, label="J delta_q")
    # plt.plot(t, seq_delta_p_2, label="delta_p")
    # plt.xlabel("t")
    # plt.ylabel("y")
    # plt.legend()
    # plt.grid(True)
    # plt.ylim([-0.01, 0.01])

    # # --------------------------------
    # plt.figure(figsize=(8, 5))
    # plt.title("dof v.s. doa")

    # indices = [0, 1, 2]
    # plt.plot(t, seq_dof[:, indices] - seq_dof[0, indices], label=[f"f_{n}" for n in indices])
    # plt.plot(t, seq_doa[:, indices] - seq_doa[0, indices], label=[f"a_{n}" for n in indices])

    # plt.xlabel("t")
    # plt.ylabel("y")
    # plt.legend()
    # plt.grid(True)

    # --------------------------------
    if method == "ours":
        plt.figure(figsize=(8, 5))
        plt.title("balance_metric")

        seq_balance_metric = np.asarray(r_data["balance_metric"])
        plt.plot(t, seq_balance_metric)

        plt.xlabel("t")
        plt.ylabel("y")
        # plt.legend()
        plt.grid(True)

    # --------------------------------
    if method == "ours":
        plt.figure(figsize=(8, 5))
        plt.title("t_ctrl_opt")

        seq_t_ctrl_opt = np.asarray(r_data["t_ctrl_opt"])
        plt.plot(t, seq_t_ctrl_opt)

        plt.xlabel("t")
        plt.ylabel("y")
        # plt.legend()
        plt.grid(True)

    # --------------------------------
    plt.figure(figsize=(8, 5))
    plt.title("obj pos")

    seq_obj_pose = np.asarray(r_data["obj_pose"])
    t_with_lift = np.arange(seq_obj_pose.shape[0]) * dt

    seq_obj_pos = seq_obj_pose[:, :3]
    plt.plot(t_with_lift, seq_obj_pos - seq_obj_pos[0, :])

    plt.xlabel("t")
    plt.ylabel("y")
    # plt.legend()
    plt.grid(True)

    # --------------------------------
    plt.figure(figsize=(8, 5))
    plt.title("normalized wrench")

    I3 = np.eye(3)
    seq_wrench = np.zeros((t.shape[0], 6))
    seq_normalized_wrench = np.zeros((t.shape[0], 6))
    seq_n_contact = np.zeros((t.shape[0]))

    for i in range(n_step):
        contacts = r_data["contacts"][i]
        n_con = len(contacts)
        seq_n_contact[i] = len(contacts)
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

    labels = ["x", "y", "z", "rx", "ry", "rz"]
    plt.plot(t, seq_normalized_wrench, label=labels)

    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    # -------------------------------
    plt.figure(figsize=(8, 5))
    plt.title("wrench")
    labels = ["x", "y", "z", "rx", "ry", "rz"]
    plt.plot(t, seq_wrench, label=labels)
    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    # -------------------------------
    plt.figure(figsize=(8, 5))
    plt.title("n_contacts")
    plt.plot(t, seq_n_contact)
    plt.xlabel("t")
    plt.ylabel("y")
    # plt.legend()
    plt.grid(True)

    # # -------------------------------
    # plt.figure(figsize=(8, 5))
    # plt.title("contact force of each fingertip")
    # seq_cf_each_finger = np.zeros((n_step, 4, 3))
    # link_names = ["rh_finger1_tip_center", "rh_finger2_tip_center", "rh_finger3_tip_center", "rh_thumb_tip_center"]

    # for i in range(n_step):
    #     contacts = r_data["contacts"][i]
    #     for contact in contacts:
    #         idx = link_names.index(contact["body1_name"])
    #         seq_cf_each_finger[i, idx, :] = contact["contact_force"]

    # for i in range(4):
    #     plt.plot(t, seq_cf_each_finger[:, i, 0], label=link_names[i])
    # plt.xlabel("t")
    # plt.ylabel("y")
    # plt.legend()
    # plt.grid(True)

    # -------------------------------
    plt.show()
