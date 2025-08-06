import sys
import os
import numpy as np
import matplotlib.pyplot as plt

parent_dir = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.append(parent_dir)

from util.robot_adaptor import RobotAdaptor
from util.pin_helper import PinocchioHelper
from util.robots.base import RobotFactory, Robot, ArmHand
from mr_utils.utils_calc import skew

if __name__ == "__main__":
    robot: ArmHand = RobotFactory.create_robot(robot_type="dummy_arm_shadow", prefix="rh")
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

    data_path = "output/debug_dummy_arm_shadow/control/core_can_60f4012b5336902b30612f5c0ef21eb8/tabletop_ur10e/scale010_pose008_0/11_grasp_ours2.npy"

    r_data = np.load(data_path, allow_pickle=True).item()

    n_step = len(r_data["dof"])

    seq_dof = np.asarray(r_data["dof"])
    seq_doa = np.asarray(r_data["doa"])
    seq_planned_dof = np.asarray(r_data["planned_dof"])

    t = np.arange(n_step) * 0.04

    # --------------------------------
    plt.figure(figsize=(8, 5))
    plt.title("contact force")

    seq_cf = []
    for i in range(n_step):
        contacts = r_data["contacts"][i]
        sum_cf = 0
        for contact in contacts:
            cf = contact["contact_force"]
            cf_mag = np.linalg.norm(cf)
            sum_cf += cf_mag
        seq_cf.append(sum_cf)
    seq_cf = np.asarray(seq_cf)

    plt.plot(t, seq_cf)

    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.ylim([0, 20])

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
    plt.title("contact model")

    seq_cf_actual = np.zeros((t.shape[0], 3))
    seq_cf_pred = np.zeros((t.shape[0], 3))
    seq_delta_p_1 = np.zeros((t.shape[0], 3))
    seq_delta_p_2 = np.zeros((t.shape[0], 3))

    I3 = np.eye(3)

    for i in range(n_step):
        contacts = r_data["contacts"][i]
        robot_adaptor.compute_jaco_a(seq_doa[i, :])
        delta_q = seq_doa[i, :] - seq_dof[i, :]
        for contact in contacts:
            body_name = contact["body1_name"]
            if body_name == "rh_thdistal":
                cf_actual = contact["contact_force"][:3]
                Ks = contact["Ks"]
                body_jaco = robot_adaptor.get_frame_jaco(body_name, type="body")

                cp_local = contact["contact_pos_local"].reshape(-1, 1)  # p_B in A
                cf_local = contact["contact_frame_local"].reshape(3, 3)  # R_B in A
                Trans = np.block([[I3, -skew(cp_local)]])
                contact_jaco = cf_local.T @ Trans @ body_jaco  # J_B in B (translation part)

                delta_p_1 = contact_jaco @ delta_q.reshape(-1, 1)
                delta_p_2 = contact["delta_p"].reshape(-1, 1)  # more accurate

                cf_pred = (Ks @ delta_p_2).reshape(-1)

                seq_cf_actual[i, :] = cf_actual
                seq_cf_pred[i, :] = cf_pred
                seq_delta_p_1[i, :] = delta_p_1.reshape(-1)
                seq_delta_p_2[i, :] = delta_p_2.reshape(-1)

    labels = ["x", "y", "z"]
    plt.plot(t, seq_cf_actual, label=labels)
    plt.plot(t, seq_cf_pred, label=[f"pred_{n}" for n in labels])

    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    # --------------------------------
    plt.figure(figsize=(8, 5))
    plt.title("delta_p v.s. J delta_q")

    plt.plot(t, seq_delta_p_1, label="J delta_q")
    plt.plot(t, seq_delta_p_2, label="delta_p")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.ylim([-0.01, 0.01])

    # --------------------------------
    plt.figure(figsize=(8, 5))
    plt.title("dof v.s. doa")

    indices = [0, 1, 2]
    plt.plot(t, seq_dof[:, indices] - seq_dof[0, indices], label=[f"f_{n}" for n in indices])
    plt.plot(t, seq_doa[:, indices] - seq_doa[0, indices], label=[f"a_{n}" for n in indices])

    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    # --------------------------------
    plt.figure(figsize=(8, 5))
    plt.title("balance_metric")

    seq_balance_metric = np.asarray(r_data["balance_metric"])
    plt.plot(t, seq_balance_metric)

    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    # --------------------------------
    plt.figure(figsize=(8, 5))
    plt.title("t_ctrl_opt")

    seq_t_ctrl_opt = np.asarray(r_data["t_ctrl_opt"])
    plt.plot(t, seq_t_ctrl_opt)

    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    # -------------------------------
    plt.show()
