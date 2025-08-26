import os
import sys
import time
import copy
import warnings
import numpy as np

from .base import BaseEval
from util.robots.base import RobotFactory, Robot, ArmHand
from util.pin_helper import PinocchioHelper
from util.robot_adaptor import RobotAdaptor
from util.grasp_controller import GraspController

warnings.filterwarnings("ignore", category=RuntimeWarning)


class tabletopDummyArmOursEval(BaseEval):
    def _initialize(self):
        self.method_name = "ours"
        robot_name = self.configs.hand_name
        robot: ArmHand = RobotFactory.create_robot(robot_type=robot_name, prefix="rh_")
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

    def _simulate_under_extforce_details(self, pregrasp_qpos, grasp_qpos, squeeze_qpos):
        # self._initialize()

        # Pre-calculated parameters
        ctrl_freq = self.ctrl_freq
        action_dt = self.action_dt
        sim_step_per_action = self.sim_step_per_action

        # initialize actuated qpos
        curr_qpos_f = self.mj_ho.get_qpos_f(names=self.robot.dof_names)
        qpos_a = self.robot_adaptor._dof2doa(curr_qpos_f)
        self.mj_ho.ctrl_qpos_a(self.robot.doa_names, qpos_a)
        init_obj_pose = self.mj_ho.get_obj_pose()

        # compute full path via interpolation
        curr_qpos_f = self.mj_ho.get_qpos_f(names=self.robot.dof_names)
        grasp_qpos_f = self._dof_data2user(grasp_qpos)
        squeeze_qpos_f = self._dof_data2user(squeeze_qpos)
        qpos_f_path_1 = self.grasp_ctrl.interplote_qpos(curr_qpos_f, grasp_qpos_f, step=ctrl_freq * 2)
        qpos_f_path_2 = self.grasp_ctrl.interplote_qpos(grasp_qpos_f, squeeze_qpos_f, step=ctrl_freq * 2)
        qpos_f_path = np.concatenate([qpos_f_path_1, qpos_f_path_2], axis=0)

        obj_mass = self.mj_ho.get_obj_mass()
        final_sum_force = self.grasp_ctrl.final_sum_force
        self.grasp_ctrl.sum_force_step = final_sum_force / qpos_f_path_2.shape[0]

        last_dq_a = np.zeros((self.robot.n_doa))
        max_steps = qpos_f_path.shape[0] * 2
        finish_init_steps = 0 * ctrl_freq
        step = 0
        waypoint_idx = 0
        stage = 1

        while step < max_steps:
            target_qpos_f = qpos_f_path[waypoint_idx]
            # get state
            curr_qpos_f = self.mj_ho.get_qpos_f(names=self.robot.dof_names)
            curr_qpos_a = self.mj_ho.get_qpos_a()
            ho_contacts = self.mj_ho.get_curr_contact_info()
            obj_pose = self.mj_ho.get_obj_pose()  # pos + quat(w,x,y,z)

            # print(f"--------------- {step} step ---------------")
            # for contact in ho_contacts:
            #     print(
            #         f"{step} step, body1_name: {contact['body1_name']}, body2_name: {contact['body2_name']}, contact_force: {contact['contact_force']}"
            #     )

            # compute some variables
            contact_force_all = np.array([contact["contact_force"][:3] for contact in ho_contacts]).reshape(-1, 3)
            curr_sum_force = np.sum(contact_force_all[:, 0])
            # print(f"curr_sum_force: {curr_sum_force}")
            grasp_matrix = self.grasp_ctrl.compute_grasp_matrix(ho_contacts)

            # terminal criteria
            if step >= qpos_f_path.shape[0] and curr_sum_force > final_sum_force:
                # print("[Info] Reached final sum force.")
                break

            t1 = time.time()
            balance_metric, _ = self.grasp_ctrl.check_wrench_balance(grasp_matrix, b_print_opt_details=False)
            t_check_balance = time.time() - t1
            # print(f"check_wrench_balance() time cost: {t_check_balance}")
            # print(f"balance_metric: {balance_metric}")

            # compute data for post-check
            if len(ho_contacts) > 0:
                ho_contacts = self.grasp_ctrl.Ks(q_a=curr_qpos_a, contacts=ho_contacts)
                for con_id, contact in enumerate(ho_contacts):
                    contact_force = contact["contact_force"][:3]
                    contact_frame = contact["contact_frame"]
                    Ks = contact["Ks"]
                    body_name = contact["body1_name"]
                    contact_pos_local = contact["contact_pos_local"]

                    self.robot_adaptor.compute_fk_a(curr_qpos_a)
                    pose_desired = self.robot_adaptor.get_frame_pose(body_name)
                    pos_desired, rot_desired = pose_desired[:3, 3].reshape(-1, 1), pose_desired[:3, :3]
                    cp_desired = rot_desired @ contact_pos_local + pos_desired

                    self.robot_adaptor.compute_fk_f(curr_qpos_f)
                    pose_actual = self.robot_adaptor.get_frame_pose(body_name)
                    pos_actual, rot_actual = pose_actual[:3, 3].reshape(-1, 1), pose_actual[:3, :3]
                    cp_actual = rot_actual @ contact_pos_local + pos_actual

                    delta_p = contact_frame.T @ (cp_desired - cp_actual).reshape(-1, 1)  # in contact local frame

                    contact["delta_p"] = delta_p
                    contact["Ks_delta_p"] = (Ks @ delta_p).reshape(-1)
                ho_contacts[con_id] = contact

            # transition from stage 1 to stage 2
            if step > finish_init_steps and balance_metric < self.grasp_ctrl.balance_thres:
                stage = 2
                # sum_force = self.grasp_ctrl.compute_desired_sum_force(
                #     grasp_matrix=grasp_matrix, obj_mass=obj_mass, b_print_opt_details=False
                # )
                # sum_force *= 2
                # final_sum_force = max(final_sum_force, sum_force)
            elif self.configs.task.control.free_stage_switch:  # if enables free_stage_switch, it allows back to stage 1
                stage = 1

            if stage == 1:
                desired_sum_force = max(0.2, curr_sum_force - self.grasp_ctrl.sum_force_step)
            elif stage == 2:
                # print("final_sum_force: ", final_sum_force)
                desired_sum_force = min(curr_sum_force, final_sum_force) + self.grasp_ctrl.sum_force_step

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
            # print(f"ctrl_opt2 time cost: {t_ctrl_opt}")

            opt_q_a = res["q_a"]
            last_dq_a = res["dq_a"]

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
