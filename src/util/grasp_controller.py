from scipy.optimize import minimize
from scipy.linalg import block_diag
import numpy as np
import time
import os
import torch
from torch.autograd.functional import jacobian as torch_jaco
from mr_utils.utils_calc import (
    isometry3dToPosQuat,
    isometry3dToPosOri,
    sciR,
    transformPositions,
    skew,
    mapping_from_space_avel_to_dquat,
)
from mr_utils.utils_torch import matrix_to_quaternion, quaternion_angular_error

try:
    from .robot_adaptor import RobotAdaptor
    from .robots.base import RobotFactory, Robot, ArmHand
except:
    from robot_adaptor import RobotAdaptor
    from robots.base import RobotFactory, Robot, ArmHand


class GraspController:
    def __init__(self, robot: ArmHand, robot_adaptor: RobotAdaptor):
        self.robot = robot
        self.robot_adaptor = robot_adaptor

        self.r_data = {
            "obj_pose": [],
            "dof": [],
            "doa": [],
            "contacts": [],
            "planned_dof": [],
            "balance_metric": [],
            "t_check_balance": [],
            "t_ctrl_opt": [],
        }

        # hyper-parameters
        self.Ke = np.diag([10000, 100, 100])  # x-axis is the contact normal
        self.Kp = np.diag(self.robot.doa_kp)
        self.Kp_inv = np.linalg.inv(self.Kp)
        self.final_sum_force = 5.0
        self.sum_force_step = 0.2

    def save_recorded_data(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.r_data, allow_pickle=True)
        print(f"Save recorded data to {path}.")

    def interplote_qpos(self, qpos1: np.array, qpos2: np.array, step: int) -> np.array:
        return np.linspace(qpos1, qpos2, step + 1)[1:]

    def Ks(self, q_a, contacts):
        """
        Per contact.
        """
        self.robot_adaptor.compute_jaco_a(q_a=q_a)
        I3 = np.eye(3)

        for i, contact in enumerate(contacts):
            body_name = contact["body1_name"]
            cp_local = contact["contact_pos_local"].reshape(-1, 1)  # p_B in A
            cf_local = contact["contact_frame_local"].reshape(3, 3)  # R_B in A
            body_jaco = self.robot_adaptor.get_frame_jaco(frame_name=body_name, type="body")  # J_A in A
            Trans = np.block([[I3, -skew(cp_local)]])
            contact_jaco = cf_local.T @ Trans @ body_jaco  # J_B in B (translation part)
            Kr_inv = contact_jaco @ self.Kp_inv @ contact_jaco.T
            Ks = np.linalg.inv(I3 + self.Ke @ Kr_inv) @ self.Ke  # in contact local frame
            contact["Ks"] = Ks
            contact["jaco"] = contact_jaco

            contacts[i] = contact

        return contacts

    def compute_grasp_matrix(self, ho_contacts) -> np.ndarray:
        n_con = len(ho_contacts)
        if n_con == 0:
            return None

        contact_frame = [contact["contact_frame"] for contact in ho_contacts]
        contact_pos_all = [contact["contact_pos"] for contact in ho_contacts]
        contact_pos_all = np.asarray(contact_pos_all).reshape(-1, 3)
        contact_centroid = contact_pos_all.mean(axis=0, keepdims=True)
        contact_r = contact_pos_all - contact_centroid
        contact_G = []
        for i in range(len(ho_contacts)):
            r = contact_r[i, :]
            n, o, t = contact_frame[i][:, 0], contact_frame[i][:, 1], contact_frame[i][:, 2]
            G = np.block(
                [
                    [n.reshape(-1, 1), o.reshape(-1, 1), t.reshape(-1, 1)],
                    [np.cross(r, n).reshape(-1, 1), np.cross(r, o).reshape(-1, 1), np.cross(r, t).reshape(-1, 1)],
                ]
            )
            contact_G.append(G)
        contact_G = np.concatenate(contact_G, axis=1)

        return contact_G

    def check_wrench_balance(self, grasp_matrix, b_print_opt_details=False):
        if grasp_matrix is None:
            return 1.0, None

        contact_G = grasp_matrix.copy()
        n_con = contact_G.shape[1] // 3

        # weights
        w_wrench = np.diag([1.0, 1, 1, 1, 1, 1])
        mu = 0.1
        gamma = 1.0

        def objective(x):
            cf = x.copy()
            wrench = contact_G @ cf.reshape(-1, 1)
            cost = wrench.T @ w_wrench @ wrench
            grad = 2 * contact_G.T @ w_wrench @ contact_G @ cf.reshape(-1, 1)
            return cost.item(), grad

        def friction_cone_constraint(x):
            cf = x.copy().reshape(-1, 3)
            constraint = mu * cf[:, 0] - np.linalg.norm(cf[:, 1:], axis=-1)  # >= 0
            return constraint.reshape(-1)

        def friction_cone_constraint_grad(x):
            cf = x.reshape(-1, 3)  # shape (n, 3)
            n = cf.shape[0]
            grad = np.zeros((n, n * 3))

            epsilon = 1e-8
            for i in range(n):
                fx, fy, fz = cf[i]
                norm_yz = np.sqrt(fy**2 + fz**2) + epsilon  # avoid divide-by-zero

                grad[i, 3 * i + 0] = mu * np.sign(fx)
                grad[i, 3 * i + 1] = -fy / norm_yz
                grad[i, 3 * i + 2] = -fz / norm_yz

            return grad

        def force_magnitude_constraint(x):
            cf = x.copy().reshape(-1, 3)
            constraint = np.sum(cf[:, 0]) - gamma  # >= 0
            return constraint.reshape(-1)

        def force_magnitude_constraint_grad(x):
            cf = x.reshape(-1, 3)  # shape (n, 3)
            grad = np.zeros_like(cf)  # shape (n, 3)
            grad[:, 0] = 1.0
            return grad.reshape(-1)  # flatten to match x shape

        constraints_list = [
            dict(type="ineq", fun=friction_cone_constraint, jac=friction_cone_constraint_grad),
            dict(type="ineq", fun=force_magnitude_constraint, jac=force_magnitude_constraint_grad),
        ]

        bounds = [(0, 10), (-5, 5), (-5, 5)] * n_con

        res = minimize(
            fun=objective,
            jac=True,
            constraints=constraints_list,
            x0=np.zeros((3 * n_con)),
            bounds=bounds,
            method="SLSQP",
            options={"ftol": 1e-6, "disp": b_print_opt_details, "maxiter": 1000},
        )

        cf = res.x.reshape(-1)
        wrench = contact_G @ cf.reshape(-1, 1)
        cost = (wrench.T @ w_wrench @ wrench).item()

        return cost, cf

    def stage2_opt4(
        self,
        stage,
        curr_q_a,
        target_q_f,
        desired_sum_force,
        ho_contacts=None,
        b_contact=True,
        b_print_opt_details=False,
    ):
        # hyper-parameters
        mu = 0.1
        dt = 0.04

        # variables for coding convenience
        n_arm_dof = self.robot.arm.n_dof
        n_hand_dof = self.robot.hand.n_dof
        n_dof = n_arm_dof + n_hand_dof
        doa2dof_matrix = self.robot_adaptor.doa2dof_matrix
        n_con = len(ho_contacts)
        curr_q_a_tensor = torch.tensor(curr_q_a)
        target_q_f_tensor = torch.tensor(target_q_f)
        joint_limits_f = self.robot_adaptor.joint_limits_f
        q_step_max = np.asarray(self.robot.doa_max_vel) * dt

        if b_contact and n_con:
            # compute grasp matrix
            contact_G_tensor = torch.from_numpy(self.compute_grasp_matrix(ho_contacts))

            # compute Ks and contact jacobian
            updated_contacts = self.Ks(q_a=curr_q_a, contacts=ho_contacts)
            Ks_all = []
            contact_jaco_all = []
            contact_force_all = []
            for i, contact in enumerate(updated_contacts):
                Ks_all.append(contact["Ks"])
                contact_jaco_all.append(contact["jaco"])
                contact_force_all.append(contact["contact_force"][:3])
            Ks_all = block_diag(*Ks_all) * 1.5  # DEBUG
            contact_jaco_all = np.concatenate(contact_jaco_all, axis=0)
            contact_force_all = np.concatenate(contact_force_all, axis=0)
            contact_jaco_all_tensor = torch.from_numpy(contact_jaco_all)
            Ks_jaco = Ks_all @ contact_jaco_all
        else:
            contact_force_all = np.zeros((0))

        # compute target hand base pose
        hand_base_name = self.robot.hand.base_name
        self.robot_adaptor.compute_fk_f(target_q_f)
        target_hb_pose = torch.tensor(self.robot_adaptor.get_frame_pose(frame_name=hand_base_name))
        target_hb_pos, target_hb_quat = target_hb_pose[:3, 3], matrix_to_quaternion(target_hb_pose[:3, :3])
        # compute current hand base jacobian (space)
        self.robot_adaptor.compute_jaco_a(curr_q_a)
        hb_jaco = self.robot_adaptor.get_frame_jaco(frame_name=hand_base_name, type="space")

        # weights
        w_q_hand = 0.1 * torch.eye(n_hand_dof, dtype=torch.float64)
        w_cp = torch.diag(torch.tensor([0.0, 100, 100], dtype=torch.float64))
        w_cp = torch.block_diag(*[w_cp for _ in range(n_con)])
        w_wrench = torch.diag(torch.tensor([1.0, 1, 1, 1, 1, 1], dtype=torch.float64))
        w_hb_pos = torch.diag(torch.tensor([0, 0, 0.0], dtype=torch.float64))
        w_hb_ori = 1.0

        def objective(x):
            var = torch.tensor(x, dtype=torch.float64, requires_grad=True)
            dq_a = var[:n_dof]
            cf = var[n_dof:]  # d contact_force
            q_a = curr_q_a_tensor + dq_a
            q_f = self.robot_adaptor._doa2dof(q_a)
            _, q_hand = q_f[:n_arm_dof], q_f[n_arm_dof:]

            # cost for hand qpos
            target_q_hand = target_q_f_tensor[n_arm_dof:]
            err_q_hand = (target_q_hand - q_hand).reshape(-1, 1)
            cost_q_hand = err_q_hand.T @ w_q_hand @ err_q_hand

            cost_wrench = 0
            cost_tangential = 0
            if b_contact and n_con > 0:
                # cost wrench
                cost_wrench = 0
                if stage == 2:
                    wrench = contact_G_tensor @ cf.reshape(-1, 1)
                    cost_wrench = wrench.T @ w_wrench @ wrench

                # cost tangential motion (restrict the tangential motion of contacts)
                dp = contact_jaco_all_tensor @ dq_a.reshape(-1, 1)
                err_cp = dp
                cost_tangential = err_cp.T @ w_cp @ err_cp

            cost_hb_pose = 0
            if stage == 1:
                self.robot_adaptor.compute_fk_a(q_a.detach().numpy())
                hb_pose = self.robot_adaptor.get_frame_pose(frame_name=hand_base_name)
                hb_pose_tensor = torch.as_tensor(hb_pose)
                hb_pos = (hb_pose_tensor[:3, 3]).requires_grad_(True)
                hb_quat = matrix_to_quaternion(hb_pose_tensor[:3, :3]).requires_grad_(True)
                # cost for hand base pose (expect lower weights for pos and higher weights for ori)
                err_hb_pos = target_hb_pos - hb_pos
                err_hb_ori = quaternion_angular_error(target_hb_quat.unsqueeze(0), hb_quat.unsqueeze(0)).squeeze()
                cost_hb_pose = err_hb_pos.T @ w_hb_pos @ err_hb_pos + w_hb_ori * err_hb_ori**2

            total_cost = cost_q_hand + cost_wrench + cost_tangential + cost_hb_pose

            # compute gradient
            total_cost.backward()
            var_grad = var.grad.numpy()  # (d cost / d var)
            grad = var_grad.reshape(-1)

            if stage == 1:
                grad_hb_pos = hb_pos.grad.numpy().reshape(1, -1)  # d (cost / d hb_pos)
                hb_pos_grad = np.matmul(grad_hb_pos, hb_jaco[:3, :])
                hb_pos_grad = np.hstack([hb_pos_grad, np.zeros((hb_pos_grad.shape[0], n_con * 3))])

                grad_hb_quat = hb_quat.grad.numpy().reshape(1, -1)  # d (cost / d hb_quat)
                hb_rot_grad = grad_hb_quat @ mapping_from_space_avel_to_dquat(hb_quat.detach().numpy()) @ hb_jaco[3:, :]
                hb_rot_grad = np.hstack([hb_rot_grad, np.zeros((hb_rot_grad.shape[0], n_con * 3))])

                grad = var_grad.reshape(-1) + hb_pos_grad.reshape(-1) + hb_rot_grad.reshape(-1)

            elif stage == 2:
                grad = var_grad.reshape(-1)

            else:
                raise NotImplementedError()

            return total_cost.item(), grad

        def contact_model_constraint(x):
            dq_a = x[:n_dof].copy()
            cf = x[n_dof:].copy()
            dcf = cf - contact_force_all

            err = dcf.reshape(-1, 1) - Ks_jaco @ dq_a.reshape(-1, 1)
            constraint = err.reshape(-1, 3)[:, 0]  # only constrain the normal forces
            return constraint.reshape(-1)  # == 0

        def contact_model_constraint_grad(x):
            idx_normal = np.arange(0, n_con * 3, 3)
            grad_cf = np.zeros((n_con, 3 * n_con))
            grad_cf[np.arange(n_con), idx_normal] = 1.0
            grad_dq_a = -Ks_jaco[idx_normal, :]

            jacobian = np.hstack([grad_dq_a, grad_cf])
            return jacobian  # shape: (n_contacts, x_dim)

        def q_limits_constraint(x):
            dq_a = x[:n_dof].copy()
            q_a = curr_q_a + dq_a
            q_f = doa2dof_matrix @ q_a.reshape(-1, 1)

            In = np.eye(n_dof)
            A = np.concatenate([-In, In], axis=0)
            b = np.concatenate([joint_limits_f[0, :], -joint_limits_f[1, :]], axis=0).reshape(-1, 1)
            constraint = A @ q_f + b
            return -constraint.reshape(-1)  # >= 0

        def q_limits_constraint_grad(x):
            grad = np.zeros((2 * n_dof, len(x)))  # shape: (2*n_dof, len(x))
            A = np.concatenate([-np.eye(n_dof), np.eye(n_dof)], axis=0)
            grad_wrt_dq_a = -A @ doa2dof_matrix
            grad[:, :n_dof] = grad_wrt_dq_a
            return grad

        def friction_cone_constraint(x):
            """
            Input normal forces must be positive.
            """
            cf = x[n_dof:].reshape(-1, 3)
            constraint = mu * cf[:, 0] - np.linalg.norm(cf[:, 1:], axis=-1)  # >= 0
            return constraint.reshape(-1)

        def friction_cone_constraint_grad(x):
            cf = x[n_dof:].reshape(-1, 3)
            fx, fy, fz = cf[:, 0], cf[:, 1], cf[:, 2]
            norm_yz = np.sqrt(fy**2 + fz**2) + 1e-8  # Avoid divide-by-zero

            grad = np.zeros((n_con, x.shape[0]))
            idx = np.arange(n_con)
            grad[idx, n_dof + 3 * idx + 0] = mu  # ∂f/∂fx
            grad[idx, n_dof + 3 * idx + 1] = -fy / norm_yz  # ∂f/∂fy
            grad[idx, n_dof + 3 * idx + 2] = -fz / norm_yz  # ∂f/∂fz

            return grad

        def force_magnitude_constraint(x):
            """
            Input normal forces must be positive.
            """
            cf = x[n_dof:].reshape(-1, 3)
            constraint = desired_sum_force - np.sum(cf[:, 0])  # == 0 / >= 0
            return constraint.reshape(-1)

        def force_magnitude_constraint_grad(x):
            n_vars = x.shape[0]
            grad = np.zeros((1, n_vars))  # shape: (1, len(x))
            idx = np.arange(n_con) * 3 + 0  # index of normal force in each contact
            grad[0, n_dof + idx] = -1.0
            return grad  # shape: (1, len(x))

        def arm_doa_constraint(x):
            dq_a_arm = x[:n_arm_dof].copy()
            constraint = dq_a_arm - 0  # == 0
            return constraint.reshape(-1)

        def arm_doa_constraint_grad(x):
            grad = np.zeros((n_arm_dof, x.shape[0]))
            grad[:, :n_arm_dof] = np.eye(n_arm_dof)
            return grad

        if stage == 1:
            if n_con == 0:
                constraints_list = [dict(type="ineq", fun=q_limits_constraint, jac=q_limits_constraint_grad)]
            else:
                constraints_list = [
                    dict(type="eq", fun=contact_model_constraint, jac=contact_model_constraint_grad),
                    dict(type="ineq", fun=q_limits_constraint, jac=q_limits_constraint_grad),
                    dict(type="ineq", fun=force_magnitude_constraint, jac=force_magnitude_constraint_grad),
                ]
        elif stage == 2:
            constraints_list = [
                dict(type="eq", fun=contact_model_constraint, jac=contact_model_constraint_grad),
                dict(type="ineq", fun=q_limits_constraint, jac=q_limits_constraint_grad),
                dict(type="ineq", fun=friction_cone_constraint, jac=friction_cone_constraint_grad),
                dict(type="eq", fun=force_magnitude_constraint, jac=force_magnitude_constraint_grad),
                dict(type="eq", fun=arm_doa_constraint, jac=arm_doa_constraint_grad),
            ]

        bounds_dq = [(-q_step_max[i], q_step_max[i]) for i in range(q_step_max.shape[0])]
        bounds_cf = [(0, 10), (-5, 5), (-5, 5)] * n_con
        bounds = bounds_dq + bounds_cf
        x0 = np.concatenate([np.zeros((n_dof)), contact_force_all], axis=0)

        res = minimize(
            fun=objective,
            jac=True,
            constraints=constraints_list,
            x0=x0,
            bounds=bounds,
            method="SLSQP",
            options={"ftol": 1e-6, "disp": b_print_opt_details, "maxiter": 1000},
        )

        res_var = res.x.reshape(-1)
        dq_a = res_var[:n_dof]
        cf = res_var[n_hand_dof:]

        res_q_a = curr_q_a + dq_a
        return res_q_a


if __name__ == "__main__":
    from pin_helper import PinocchioHelper

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

    grasp_ctrl = GraspController(robot=robot, robot_adaptor=robot_adaptor)

    # curr_q_a = (robot_adaptor.joint_limits_f[0, :] + robot_adaptor.joint_limits_f[1, :]) / 2.0
    # target_q_f = robot_adaptor._doa2dof(curr_q_a) + 0.01

    # t1 = time.time()
    # grasp_ctrl.stage1_opt(curr_q_a, target_q_f)
    # print(f"time cost: {time.time() - t1}")
