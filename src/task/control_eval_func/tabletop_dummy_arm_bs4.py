import os
import sys
import time
import copy
import warnings
import numpy as np
from scipy.linalg import block_diag

from .base import BaseEval
from util.robots.base import RobotFactory, Robot, ArmHand
from util.pin_helper import PinocchioHelper
from util.robot_adaptor import RobotAdaptor
from util.grasp_controller import GraspController

warnings.filterwarnings("ignore", category=RuntimeWarning)


class tabletopDummyArmBS4Eval(BaseEval):
    def _initialize(self):
        self.method_name = "bs4"
        robot_name = self.configs.hand_name
        robot_prefix = "rh_" if "allegro" not in robot_name else ""
        robot: ArmHand = RobotFactory.create_robot(robot_type=robot_name, prefix=robot_prefix)

        robot_file_path = robot.get_file_path("mjcf")
        dof_names = robot.dof_names
        doa_names = robot.doa_names
        doa2dof_matrix = robot.doa2dof_matrix

        self.robot = robot
        self.robot_model = PinocchioHelper(robot_file_path=robot_file_path, robot_file_type="mjcf")
        self.robot_adaptor = RobotAdaptor(
            robot_model=self.robot_model,
            dof_names=dof_names,
            doa_names=doa_names,
            doa2dof_matrix=doa2dof_matrix,
        )
        self.grasp_ctrl = GraspController(robot=self.robot, robot_adaptor=self.robot_adaptor)
        self.dof_data2user_indices = [self.grasp_data["joint_names"].index(name) for name in dof_names]

    def _dof_data2user(self, q):
        return q[..., self.dof_data2user_indices].copy()

    def damped_pinv(self, J):
        lambd = 1.0
        I_mat = np.eye(J.shape[0])
        J_inv = J.T @ np.linalg.inv(J @ J.T + lambd**2 * I_mat)
        return J_inv

    def _simulate_under_extforce_details(self, pregrasp_qpos, grasp_qpos, squeeze_qpos):
        # self._initialize()

        # Pre-calculated parameters
        ctrl_freq = self.ctrl_freq
        action_dt = self.action_dt
        sim_step_per_action = self.sim_step_per_action
        b_debug = self.configs.task.debug_viewer
        arm_ndoa = self.robot.arm.n_doa
        hand_ndoa = self.robot.hand.n_doa

        # initialize actuated qpos
        curr_qpos_f = self.mj_ho.get_qpos_f(names=self.robot.dof_names)
        qpos_a = self.robot_adaptor._dof2doa(curr_qpos_f)
        self.mj_ho.ctrl_qpos_a(self.robot.doa_names, qpos_a)

        # compute full path via interpolation
        curr_qpos_f = self.mj_ho.get_qpos_f(names=self.robot.dof_names)
        grasp_qpos_f = self._dof_data2user(grasp_qpos)
        squeeze_qpos_f = self._dof_data2user(squeeze_qpos)
        qpos_f_path_1 = self.grasp_ctrl.interplote_qpos(curr_qpos_f, grasp_qpos_f, step=ctrl_freq * 2)
        qpos_f_path_2 = self.grasp_ctrl.interplote_qpos(grasp_qpos_f, squeeze_qpos_f, step=ctrl_freq * 2)
        qpos_f_path = np.concatenate([qpos_f_path_1, qpos_f_path_2], axis=0)

        final_sum_force = self.grasp_ctrl.final_sum_force
        force_incre_step = final_sum_force / qpos_f_path_2.shape[0]
        last_dq_a = np.zeros((self.robot.n_doa))
        max_steps = qpos_f_path.shape[0] * 2

        stage = 1
        step = 0
        waypoint_idx = 0

        while step < max_steps:
            target_qpos_f = qpos_f_path[waypoint_idx]
            # get state
            curr_qpos_f = self.mj_ho.get_qpos_f(names=self.robot.dof_names)
            curr_qpos_a = self.mj_ho.get_qpos_a()
            ho_contacts = self.mj_ho.get_curr_contact_info()
            obj_pose = self.mj_ho.get_obj_pose()  # pos + quat(w,x,y,z)

            # compute some variables
            n_con = len(ho_contacts)
            contact_force_all = np.array([contact["contact_force"][:3] for contact in ho_contacts]).reshape(-1, 3)
            curr_sum_force = np.sum(contact_force_all[:, 0])
            grasp_matrix = self.grasp_ctrl.compute_grasp_matrix(ho_contacts)

            # terminal criteria
            if step >= qpos_f_path.shape[0] and curr_sum_force > final_sum_force:
                break

            t1 = time.time()
            balance_metric, _ = self.grasp_ctrl.check_wrench_balance(grasp_matrix, b_print_opt_details=False)
            t_check_balance = time.time() - t1

            if b_debug:
                print(f"--------------- {step} step ---------------")
                for contact in ho_contacts:
                    print(
                        f"{step} step, body1_name: {contact['body1_name']}, body2_name: {contact['body2_name']}, "
                        + f"contact_force: {contact['contact_force']}"
                    )
                print(f"curr_sum_force: {curr_sum_force}")
                print(f"check_wrench_balance() time cost: {t_check_balance}")
                print(f"balance_metric: {balance_metric}")

            # transition from stage 1 to stage 2
            if balance_metric < self.grasp_ctrl.balance_thres or (
                self.configs.task.control.stage2_after_full_path and step > qpos_f_path.shape[0]
            ):
                stage = 2
            elif self.configs.task.control.free_stage_switch:  # if enables free_stage_switch, it allows back to stage 1
                stage = 1

            if stage == 1:
                desired_sum_force = max(self.grasp_ctrl.stage1_force_thres, curr_sum_force - force_incre_step)
                t1 = time.time()
                res = self.grasp_ctrl.ctrl_opt3(
                    stage=stage,
                    dt=action_dt,
                    curr_q_a=curr_qpos_a,
                    target_q_f=target_qpos_f,
                    desired_sum_force=desired_sum_force,
                    last_dq_a=last_dq_a,
                    ho_contacts=ho_contacts,
                    grasp_matrix=grasp_matrix,
                    b_contact=True,
                    b_print_opt_details=False,
                )
                t_ctrl_opt = time.time() - t1
                opt_q_a = res["q_a"]
                last_dq_a = res["dq_a"]

            elif stage == 2:
                t1 = time.time()
                # compute qpos error
                curr_arm_qpos_a = curr_qpos_a[:arm_ndoa]
                curr_hand_qpos_a = curr_qpos_a[-hand_ndoa:]
                target_qpos_a = self.robot_adaptor._dof2doa(target_qpos_f)
                target_hand_qpos_a = target_qpos_a[-hand_ndoa:]
                hand_qpos_err = (target_hand_qpos_a - curr_hand_qpos_a).reshape(-1, 1)
                w_q = np.ones_like(curr_hand_qpos_a)

                updated_contacts = self.grasp_ctrl.Ks(curr_qpos_a, ho_contacts)
                desired_sum_force = min(curr_sum_force, final_sum_force) + force_incre_step
                if n_con > 0:
                    res = self.grasp_ctrl.ctrl_opt_bs4(
                        stage=stage,
                        dt=action_dt,
                        curr_q_a=curr_qpos_a,
                        target_q_f=target_qpos_f,
                        desired_sum_force=desired_sum_force,
                        last_dq_a=last_dq_a,
                        ho_contacts=ho_contacts,
                        grasp_matrix=grasp_matrix,
                        b_contact=True,
                        b_print_opt_details=b_debug,
                    )
                    opt_cf = res["cf"]

                    Ks_all = []
                    contact_jaco_all = []
                    contact_force_all = []
                    for _, contact in enumerate(updated_contacts):
                        Ks_all.append(contact["Ks_h"])
                        contact_jaco_all.append(contact["jaco_h"])
                        contact_force_all.append(contact["contact_force"][:3])
                    Ks_all = block_diag(*Ks_all)
                    contact_jaco_all = np.concatenate(contact_jaco_all, axis=0)
                    contact_force_all = np.concatenate(contact_force_all, axis=0).reshape(-1, 1)
                    Ks_jaco = Ks_all @ contact_jaco_all

                    target_contact_force_all = opt_cf.reshape(-1, 1)
                    contact_force_err = target_contact_force_all - contact_force_all
                    in_contact_q_indices = np.any(contact_jaco_all != 0, axis=0)
                    w_q[in_contact_q_indices] = 0
                    if b_debug:
                        print(f"in_contact_q_indices: {in_contact_q_indices}")

                w_q = np.diag(w_q)
                gain_q = 1.0 * np.eye(hand_ndoa)  # 1.0

                delta_hand_q_a = w_q @ gain_q @ hand_qpos_err
                if n_con:
                    Kp_inv = self.grasp_ctrl.Kp_inv[-hand_ndoa:, -hand_ndoa:]
                    if b_debug:
                        print("Ks_jaco @ Kp_inv_JT: ", np.diag(Ks_jaco @ Kp_inv @ contact_jaco_all.T))
                    force_control_input = 1.0 * self.damped_pinv(Ks_jaco) @ contact_force_err
                    # force_control_input = 1.0 * Kp_inv @ contact_jaco_all.T @ contact_force_err

                    delta_hand_q_a += force_control_input
                    # if b_debug:
                    #     print(f"pos_control_input: {(w_q @ gain_q @ hand_qpos_err).reshape(-1)}")
                    #     print(f"force_control_input: {force_control_input.reshape(-1)}")

                delta_hand_q_a = delta_hand_q_a.reshape(-1)
                opt_hand_q_a = curr_hand_qpos_a + delta_hand_q_a
                opt_q_a = np.concatenate([curr_arm_qpos_a, opt_hand_q_a], axis=0)
                last_dq_a = np.concatenate([np.zeros_like(curr_arm_qpos_a), delta_hand_q_a], axis=0)
                t_ctrl_opt = time.time() - t1

            assert sim_step_per_action % 5 == 0
            self.mj_ho.ctrl_qpos_a_with_interp(
                curr_qpos_a, opt_q_a, names=self.robot.doa_names, step_outer=sim_step_per_action // 5, step_inner=5
            )

            step += 1  # next step
            waypoint_idx = min(waypoint_idx + 1, len(qpos_f_path) - 1)

            self.grasp_ctrl.r_data["obj_pose"].append(obj_pose)
            self.grasp_ctrl.r_data["dof"].append(curr_qpos_f)
            self.grasp_ctrl.r_data["doa"].append(curr_qpos_a)
            self.grasp_ctrl.r_data["contacts"].append(ho_contacts)
            self.grasp_ctrl.r_data["planned_dof"].append(target_qpos_f)
            self.grasp_ctrl.r_data["balance_metric"].append(balance_metric)
            self.grasp_ctrl.r_data["t_check_balance"].append(t_check_balance)
            self.grasp_ctrl.r_data["t_ctrl_opt"].append(t_ctrl_opt)

        return
