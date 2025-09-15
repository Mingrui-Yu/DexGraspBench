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
    config.use_multi_contact_model = True
    grasp_ctrl = GraspController(config, robot, robot_adaptor)

    method = "ours"
    data_path = "output/learn_dummy_arm_shadow/cases/for_overview/partial_pc_02_2_dist_2_pos_6.npy"

    r_data = np.load(data_path, allow_pickle=True).item()

    n_step = len(r_data["dof"])

    seq_dof = np.asarray(r_data["dof"])
    seq_doa = np.asarray(r_data["doa"])
    seq_planned_dof = np.asarray(r_data["planned_dof"])

    dt = 0.1
    t = np.arange(n_step) * dt

    # --------------------------------

    seq_cf_actual = np.zeros((t.shape[0], 3))
    seq_cf_pred = np.zeros((t.shape[0], 3))
    seq_delta_p_1 = np.zeros((t.shape[0], 3))
    seq_delta_p_2 = np.zeros((t.shape[0], 3))
    seq_d_cf_actual = np.zeros((t.shape[0], 3))
    seq_d_cf_pred = np.zeros((t.shape[0], 3))

    seq_d_qa = np.zeros((t.shape[0], seq_doa.shape[1]))
    seq_d_delta_q = np.zeros((t.shape[0], seq_doa.shape[1]))
    seq_delta_q = np.zeros((t.shape[0], seq_doa.shape[1]))

    n_arm_dof = 6
    I3 = np.eye(3)
    body_name = "rh_thdistal"
    # body_name = "rh_ffdistal"

    for i in range(n_step):
        contacts = r_data["contacts"][i]
        robot_adaptor.compute_jaco_a(seq_doa[i, :])
        delta_q = seq_doa[i, :] - seq_dof[i, :]
        seq_delta_q[i, :] = delta_q.copy()

        if i > 0:
            d_delta_q = seq_delta_q[i, :] - seq_delta_q[i - 1, :]
            d_qa = seq_doa[i, :] - seq_doa[i - 1, :]
            seq_d_qa[i, :] = d_qa
            seq_d_delta_q[i, :] = d_delta_q

        if len(contacts) > 0:
            _, stacked = grasp_ctrl.Ks(q_a=seq_doa[i, :], q_f=seq_dof[i, :], contacts=contacts)
            contact_force_all = np.concatenate([contact["contact_force"][:3] for contact in contacts], axis=0)

            contact_jaco = stacked["jaco_a"]
            contact_jaco[:, :n_arm_dof] = 0
            Ks = stacked["Ks_h"]
            cf_pred = (Ks @ contact_jaco @ delta_q.reshape(-1, 1)).reshape(-1)

            for idx, contact in enumerate(contacts):
                if body_name == contact["body1_name"]:
                    seq_cf_actual[i, :] += contact_force_all.reshape(-1, 3)[idx, :]
                    seq_cf_pred[i, :] += cf_pred.reshape(-1, 3)[idx, :]
                    # seq_delta_p_1[i, :] = delta_p_1.reshape(-1, 3)[idx, :]

                    if i > 0:
                        # velocity-level contact model
                        d_cf_actual = seq_cf_actual[i, :] - seq_cf_actual[i - 1, :]
                        seq_d_cf_actual[i, :] = d_cf_actual
                        d_cf_pred = Ks @ contact_jaco @ d_delta_q.reshape(-1, 1)
                        # d_cf_pred = Ks @ contact_jaco @ d_qa.reshape(-1, 1)
                        seq_d_cf_pred[i, :] += d_cf_pred.reshape(-1, 3)[idx, :]  # notice the '+'

    # --------------------------------
    # position-level contact model
    plt.figure(figsize=(8, 5))
    plt.title("position-level contact model")
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

    # --------------------------------
    # velocity-level contact model
    plt.figure(figsize=(8, 5))
    plt.title("velocity-level contact model")

    for i, n in enumerate(labels):
        # 画实线 (actual)
        plt.plot(t, seq_d_cf_actual[:, i], label=n)
        # 画虚线 (pred)，颜色和前一条线一致
        plt.plot(t, seq_d_cf_pred[:, i], linestyle="--", color=plt.gca().lines[-1].get_color(), label=f"pred_{n}")

    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    plt.show()
