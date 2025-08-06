import os
import sys
import time

import numpy as np

from .base import BaseEval
from util.robots.base import RobotFactory, Robot, ArmHand
from util.pin_helper import PinocchioHelper
from util.robot_adaptor import RobotAdaptor
from util.grasp_controller import GraspController


class tabletopDummyArmOursEval(BaseEval):
    def _initialize(self):
        robot_name = self.configs.hand_name
        robot: ArmHand = RobotFactory.create_robot(robot_type=robot_name, prefix="rh")
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

    def _simulate_under_extforce_details(self, pre_obj_qpos):
        self._initialize()

        # 1. Set object gravity
        external_force_direction = np.array([0.0, 0, -1, 0, 0, 0])
        self.mj_ho.set_ext_force_on_obj(10 * external_force_direction * self.configs.task.obj_mass)

        # --------------------------------------------------------------------

        # initialize actuated qpos
        curr_qpos_f = self.mj_ho.get_qpos_f(names=self.robot.dof_names)
        qpos_a = self.robot_adaptor._dof2doa(curr_qpos_f)
        self.mj_ho.ctrl_qpos_a(self.robot.doa_names, qpos_a)

        # compute full path via interpolation
        curr_qpos_f = self.mj_ho.get_qpos_f(names=self.robot.dof_names)
        grasp_qpos_f = self._dof_data2user(self.grasp_data["grasp_qpos"])
        squeeze_qpos_f = self._dof_data2user(self.grasp_data["squeeze_qpos"])
        qpos_f_path_1 = self.grasp_ctrl.interplote_qpos(curr_qpos_f, grasp_qpos_f, step=25 * 2)
        qpos_f_path_2 = self.grasp_ctrl.interplote_qpos(grasp_qpos_f, squeeze_qpos_f, step=25 * 1)
        qpos_f_path = np.concatenate([qpos_f_path_1, qpos_f_path_2], axis=0)

        i = 0
        while True:
            target_qpos_f = qpos_f_path[i]
            # get state
            curr_qpos_f = self.mj_ho.get_qpos_f(names=self.robot.dof_names)
            curr_qpos_a = self.mj_ho.get_qpos_a()
            ho_contacts = self.mj_ho.get_curr_contact_info()
            obj_pose = self.mj_ho.get_obj_pose()  # pos + quat(w,x,y,z)

            print(f"--------------- {i} step ---------------")
            for contact in ho_contacts:
                print(
                    f"{i} step, body1_name: {contact['body1_name']}, body2_name: {contact['body2_name']}, contact_force: {contact['contact_force']}"
                )

            # compute some variables
            contact_force_all = np.array([contact["contact_force"][:3] for contact in ho_contacts]).reshape(-1, 3)
            curr_sum_force = np.sum(contact_force_all[:, 0])
            print(f"curr_sum_force: {curr_sum_force}")
            grasp_matrix = self.grasp_ctrl.compute_grasp_matrix(ho_contacts)

            # terminal criteria
            if curr_sum_force > self.grasp_ctrl.final_sum_force:
                print("[Info] Reached final sum force.")
                break

            t1 = time.time()
            balance_metric, _ = self.grasp_ctrl.check_wrench_balance(grasp_matrix, b_print_opt_details=False)
            t_check_balance = time.time() - t1
            print(f"check_wrench_balance() time cost: {t_check_balance}")
            print(f"balance_metric: {balance_metric}")

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

            if balance_metric > 0.3:
                desired_sum_force = 0.3

                t1 = time.time()
                opt_q_a = self.grasp_ctrl.stage2_opt4(
                    stage=1,
                    curr_q_a=curr_qpos_a,
                    target_q_f=target_qpos_f,
                    desired_sum_force=desired_sum_force,
                    ho_contacts=ho_contacts,
                    b_contact=True,
                    b_print_opt_details=True,
                )
                t_ctrl_opt = time.time() - t1
                print(f"stage2_opt4 time cost: {t_ctrl_opt}")
            else:
                desired_sum_force = curr_sum_force + self.grasp_ctrl.sum_force_step

                t1 = time.time()
                opt_q_a = self.grasp_ctrl.stage2_opt4(
                    stage=2,
                    curr_q_a=curr_qpos_a,
                    target_q_f=target_qpos_f,
                    desired_sum_force=desired_sum_force,
                    ho_contacts=ho_contacts,
                    b_contact=True,
                    b_print_opt_details=True,
                )
                t_ctrl_opt = time.time() - t1
                print(f"stage2_opt4 time cost: {t_ctrl_opt}")

            # send control command
            self.mj_ho.ctrl_qpos_a(names=self.robot.doa_names, q_a=opt_q_a)
            self.mj_ho.control_hand_step(step_inner=10)

            i = min(i + 1, len(qpos_f_path) - 1)  # next step

            self.grasp_ctrl.r_data["obj_pose"].append(obj_pose)
            self.grasp_ctrl.r_data["dof"].append(curr_qpos_f)
            self.grasp_ctrl.r_data["doa"].append(curr_qpos_a)
            self.grasp_ctrl.r_data["contacts"].append(ho_contacts)
            self.grasp_ctrl.r_data["planned_dof"].append(target_qpos_f)
            self.grasp_ctrl.r_data["balance_metric"].append(balance_metric)
            self.grasp_ctrl.r_data["t_check_balance"].append(t_check_balance)
            self.grasp_ctrl.r_data["t_ctrl_opt"].append(t_ctrl_opt)

        # 6. Lift the object
        curr_qpos_a = self.mj_ho.get_qpos_a()
        lift_qpos_a = curr_qpos_a.copy()
        lift_qpos_a[2] += 0.2  # lift, by IK

        path = self.grasp_ctrl.interplote_qpos(curr_qpos_a, lift_qpos_a, step=25 * 2)
        for q_a in path:
            self.mj_ho.ctrl_qpos_a(names=self.robot.doa_names, q_a=q_a)
            self.mj_ho.control_hand_step(step_inner=10)

        # save recorded data
        method_name = "ours2"
        save_path = self.input_npy_path.replace("graspdata", "control").replace(".npy", f"_{method_name}.npy")
        self.grasp_ctrl.save_recorded_data(path=save_path)

        return
