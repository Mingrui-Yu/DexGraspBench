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
        self.Ke = np.diag([10000, 1000, 1000])  # x-axis is the contact normal
        self.Kp = np.diag(self.robot.doa_kp)
        self.Kp_inv = np.linalg.inv(self.Kp)
        self.final_sum_force = 15.0
        self.sum_force_step = 1.0
        self.balance_thres = 0.4
        self.mu = 0.3  # friction coef

    def save_recorded_data(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.r_data, allow_pickle=True)
        print(f"Save recorded control data to {path}.")

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
        contact_r = contact_r * 100.0  # unit from m to cm; then, the unit of torque is (N x cm)

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

    def compute_normalized_wrench(self, grasp_matrix, contact_forces):
        wrench = (grasp_matrix @ contact_forces.reshape(-1, 1)).reshape(-1)

        cf = contact_forces.reshape(-1, 3)
        G = grasp_matrix.reshape(6, -1, 3).transpose(1, 0, 2)
        fts = np.matmul(G, cf[:, :, None]).reshape(-1, 6)
        sum_forces_mag = np.sum(np.linalg.norm(fts[:, :3], axis=1))
        sum_torques_mag = np.sum(np.linalg.norm(fts[:, 3:], axis=1))

        wrench[:3] /= sum_forces_mag + 1e-8
        wrench[3:] /= sum_torques_mag + 1e-8
        return wrench

    def check_wrench_balance(self, grasp_matrix, b_print_opt_details=False):
        if grasp_matrix is None:
            return 1.0, None

        contact_G = grasp_matrix.copy()
        n_con = contact_G.shape[1] // 3

        if n_con < 2:  # only one contact cannot be in wrench balance
            return 1.0, None

        # weights
        w_wrench = np.diag([1.0, 1, 1, 1, 1, 1])
        mu = self.mu
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

        bounds = [(0, 10), (-10, 10), (-10, 10)] * n_con

        res = minimize(
            fun=objective,
            jac=True,
            constraints=constraints_list,
            x0=np.zeros((3 * n_con)),
            bounds=bounds,
            method="SLSQP",
            options={"ftol": 1e-6, "disp": b_print_opt_details, "maxiter": 200},
        )

        cf = res.x.reshape(-1)

        normalized_wrench = self.compute_normalized_wrench(contact_G, cf)
        metric = np.linalg.norm(normalized_wrench)

        return metric, cf

    def compute_desired_sum_force(self, grasp_matrix, obj_mass, b_print_opt_details=False):
        if grasp_matrix is None:
            return 1.0, None

        contact_G = grasp_matrix.copy()
        n_con = contact_G.shape[1] // 3
        # gravity = 1.5 * obj_mass * np.array([0, 0, -9.81, 0, 0, 0]).reshape(-1, 1)

        if n_con < 2:  # only one contact cannot be in wrench balance
            return 1.0, None

        # weights
        w_wrench = np.diag([1.0, 1, 1, 1, 1, 1])
        mu = self.mu
        # gamma = 1.0

        def objective(x):
            cf = x.copy()
            gravity = -1 * obj_mass * 9.81 * self.gravity_direction.reshape(-1, 1)
            wrench = contact_G @ cf.reshape(-1, 1) + gravity

            # G = contact_G.reshape(6, -1, 3).transpose(1, 0, 2)
            # fts = np.matmul(G, cf.reshape(-1, 3)[:, :, None]).reshape(-1, 6)
            # sum_forces_mag = np.sum(np.linalg.norm(fts[:, :3], axis=1))
            # sum_torques_mag = np.sum(np.linalg.norm(fts[:, 3:], axis=1))
            # wrench[:3] /= sum_forces_mag + 1e-8
            # wrench[3:] /= sum_torques_mag + 1e-8

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

        def wrench_constraint(x):
            cf = x.copy()
            gravity = -1 * obj_mass * 9.81 * self.gravity_direction.reshape(-1, 1)
            wrench = contact_G @ cf.reshape(-1, 1) + gravity
            constraint = wrench[self.gravity_direction_idx] - 0  #  >= 0
            return constraint.reshape(-1)

        constraints_list = [
            dict(type="ineq", fun=friction_cone_constraint, jac=friction_cone_constraint_grad),
            # dict(type="ineq", fun=force_magnitude_constraint, jac=force_magnitude_constraint_grad),
            dict(type="ineq", fun=wrench_constraint),
        ]

        bounds = [(0, 10), (-10, 10), (-10, 10)] * n_con

        cf_all = []

        g_lst = [2]

        for i in g_lst:
            self.gravity_direction = np.zeros((6))
            self.gravity_direction_idx = i
            self.gravity_direction[i] = 1

            res = minimize(
                fun=objective,
                jac=True,
                constraints=constraints_list,
                x0=np.zeros((3 * n_con)),
                bounds=bounds,
                method="SLSQP",
                options={"ftol": 1e-6, "disp": b_print_opt_details, "maxiter": 200},
            )
            cf = res.x.reshape(-1)
            cf_all.append(cf.reshape(-1, 3))

        cf_all = np.array(cf_all).reshape(len(g_lst), -1, 3)
        cf_all_normal = cf_all[:, :, 0]
        cf_max = cf_all_normal.max(axis=0)
        sum_normal_force = np.sum(cf_max)

        # normalized_wrench = self.compute_normalized_wrench(contact_G, cf)
        # metric = np.linalg.norm(normalized_wrench)
        # sum_normal_force = np.sum(cf.reshape(-1, 3)[:, 0])

        return sum_normal_force

    def ctrl_opt3(
        self,
        stage,
        dt,
        curr_q_a,
        target_q_f,
        desired_sum_force,
        last_dq_a,
        ho_contacts=None,
        grasp_matrix=None,
        b_contact=True,
        b_print_opt_details=False,
    ):
        # hyper-parameters
        mu = self.mu

        # variables for coding convenience
        n_arm_dof = self.robot.arm.n_dof
        n_hand_dof = self.robot.hand.n_dof
        n_dof = n_arm_dof + n_hand_dof
        doa2dof_matrix = self.robot_adaptor.doa2dof_matrix
        n_con = len(ho_contacts)
        joint_limits_f = self.robot_adaptor.joint_limits_f
        q_step_max = np.asarray(self.robot.doa_max_vel) * dt

        if b_contact and n_con:
            # compute grasp matrix
            contact_G = self.compute_grasp_matrix(ho_contacts) if grasp_matrix is None else grasp_matrix

            # compute Ks and contact jacobian
            updated_contacts = self.Ks(q_a=curr_q_a, contacts=ho_contacts)
            Ks_all = []
            contact_jaco_all = []
            contact_force_all = []
            for i, contact in enumerate(updated_contacts):
                Ks_all.append(contact["Ks"])
                contact_jaco_all.append(contact["jaco"])
                contact_force_all.append(contact["contact_force"][:3])
            Ks_all = block_diag(*Ks_all)
            contact_jaco_all = np.concatenate(contact_jaco_all, axis=0)
            contact_force_all = np.concatenate(contact_force_all, axis=0)
            Ks_jaco = Ks_all @ contact_jaco_all
        else:
            contact_force_all = np.zeros((0))

        # compute target hand base pose
        hand_base_name = self.robot.hand.base_name
        self.robot_adaptor.compute_fk_f(target_q_f)
        target_hb_pose = self.robot_adaptor.get_frame_pose(frame_name=hand_base_name)
        target_hb_pos, target_hb_ori = isometry3dToPosOri(target_hb_pose)

        # weights
        w_hb_pose = np.diag([0, 0, 100.0, 10.0, 10.0, 10.0])
        w_q_hand = 1.0 * np.eye(n_hand_dof)
        w_dqa = 0.001 * np.eye(n_dof)
        w_ddqa = [0.00001] * n_arm_dof + [0.001] * n_hand_dof
        w_ddqa = np.diag(w_ddqa)
        w_cp = np.diag([0.0, 100, 100])
        w_cp = block_diag(*[w_cp for _ in range(n_con)])
        w_cf = np.diag([0.0, 0.1, 0.1])
        w_cf = block_diag(*[w_cf for _ in range(n_con)])
        w_wrench = np.diag([1.0, 1, 1, 1, 1, 1])

        def objective(x):
            dq_a = x[:n_dof].copy()
            cf = x[n_dof:].copy()  # d contact_force
            q_a = curr_q_a + dq_a
            q_f = doa2dof_matrix @ q_a
            _, q_hand = q_f[:n_arm_dof], q_f[n_arm_dof:]

            # cost for hand qpos
            target_q_hand = target_q_f[n_arm_dof:]
            self.err_q_hand = err_q_hand = (q_hand - target_q_hand).reshape(-1, 1)
            cost_q_hand = err_q_hand.T @ w_q_hand @ err_q_hand

            # cost for dqa
            self.err_dqa = err_dqa = (dq_a / dt).reshape(-1, 1)
            cost_dqa = err_dqa.T @ w_dqa @ err_dqa

            # cost for ddqa
            self.err_ddqa = err_ddqa = ((dq_a - last_dq_a) / dt**2).reshape(-1, 1)
            cost_ddqa = err_ddqa.T @ w_ddqa @ err_ddqa

            cost_wrench = 0
            cost_tan_motion = 0
            cost_tan_cf = 0
            if b_contact and n_con > 0:
                if stage == 1:
                    # cost tangential motion (restrict the tangential motion of contacts)
                    dp = contact_jaco_all @ dq_a.reshape(-1, 1)
                    self.err_cp = err_cp = dp
                    cost_tan_motion = err_cp.T @ w_cp @ err_cp

                if stage == 2:
                    # cost wrench
                    self.wrench = wrench = contact_G @ cf.reshape(-1, 1)
                    cost_wrench = wrench.T @ w_wrench @ wrench

                    # cost tangential force
                    dcf = Ks_jaco @ dq_a.reshape(-1, 1)
                    pred_next_cf = contact_force_all.reshape(-1, 1) + dcf
                    self.err_cf = err_cf = cf.reshape(-1, 1) - pred_next_cf
                    cost_tan_cf = err_cf.T @ w_cf @ err_cf

            cost_hb_pose = 0
            if stage == 1:
                self.robot_adaptor.compute_fk_a(q_a)
                hb_pose = self.robot_adaptor.get_frame_pose(frame_name=hand_base_name)
                hb_pos, hb_quat = isometry3dToPosQuat(hb_pose)

                err_hb_pos = hb_pos - target_hb_pos
                hb_ori = sciR.from_quat(hb_quat)
                err_hb_ori = (hb_ori * target_hb_ori.inv()).as_rotvec()
                self.err_hb_pose = err_hb_pose = np.concatenate([err_hb_pos, err_hb_ori], axis=0).reshape(-1, 1)  # 6D
                cost_hb_pose = err_hb_pose.T @ w_hb_pose @ err_hb_pose

            total_cost = cost_dqa + cost_ddqa + cost_q_hand + cost_wrench + cost_tan_motion + cost_tan_cf + cost_hb_pose
            return total_cost.item()

        def jacobian(x):
            dq_a = x[:n_dof].copy()
            q_a = curr_q_a + dq_a
            grad = np.zeros(x.shape[0])

            # grad of cost_dqa
            err_dqa = self.err_dqa
            grad_dqa = 2.0 / dt * w_dqa @ err_dqa
            grad[:n_dof] += grad_dqa.reshape(-1)

            # grad of cost_ddqa
            err_ddqa = self.err_ddqa
            grad_dqa = 2.0 / dt**2 * w_ddqa @ err_ddqa
            grad[:n_dof] += grad_dqa.reshape(-1)

            # grad of cost_q_hand
            J_qhand_dqa = doa2dof_matrix[n_arm_dof:, :]  # (n_hand_dof, n_dof)
            err_q_hand = self.err_q_hand
            grad_dqa = 2.0 * (J_qhand_dqa.T @ w_q_hand @ err_q_hand).reshape(-1)  # shape: (n_dof,)
            grad[:n_dof] += grad_dqa.reshape(-1)

            if b_contact and n_con > 0:
                if stage == 1:
                    # grad of cost_tan_motion
                    err_cp = self.err_cp
                    grad_dqa = 2.0 * (contact_jaco_all.T @ w_cp @ err_cp).reshape(-1)  # shape (n_dof,)
                    grad[:n_dof] += grad_dqa.reshape(-1)

                if stage == 2:
                    # grad of cost_wrench
                    wrench = self.wrench
                    grad_cf = 2 * (contact_G.T @ w_wrench @ wrench)  # shape (n, 1)
                    grad[n_dof:] += grad_cf.reshape(-1)
                    # grad of cost_tan_cf
                    err_cf = self.err_cf
                    grad_dqa = -2 * Ks_jaco.T @ w_cf @ err_cf  # shape: (n_dof, 1)
                    grad_cf = 2 * w_cf @ err_cf  # shape: (n_con * 3, 1)
                    grad[:n_dof] += grad_dqa.reshape(-1)
                    grad[n_dof:] += grad_cf.reshape(-1)

            if stage == 1:
                # grad of cost_hb_pose
                self.robot_adaptor.compute_jaco_a(q_a)
                hb_jaco = self.robot_adaptor.get_frame_jaco(frame_name=hand_base_name, type="space")
                err_hb_pose = self.err_hb_pose
                grad_dqa = 2.0 * hb_jaco.T @ w_hb_pose @ err_hb_pose
                grad[:n_dof] += grad_dqa.reshape(-1)

            return grad

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

        def check_gradient(objective_fn, x0, epsilon=1e-6, verbose=True, atol=1e-5, rtol=1e-3):
            """
            Compare analytical vs numerical gradients of a given objective function at x0.

            Parameters:
                objective_fn (function): Function that returns cost and gradient: cost, grad = objective_fn(x)
                x0 (np.ndarray): Point at which to check the gradient
                epsilon (float): Step size for finite difference
                verbose (bool): Whether to print differences
                atol (float): Absolute tolerance for error
                rtol (float): Relative tolerance for error
            """
            cost, grad_analytical = objective_fn(x0)
            grad_numerical = np.zeros_like(x0)

            for i in range(len(x0)):
                x_forward = x0.copy()
                x_backward = x0.copy()
                x_forward[i] += epsilon
                x_backward[i] -= epsilon
                f_forward, _ = objective_fn(x_forward)
                f_backward, _ = objective_fn(x_backward)
                grad_numerical[i] = (f_forward - f_backward) / (2 * epsilon)

            # Compute difference
            abs_diff = np.abs(grad_analytical - grad_numerical)
            rel_diff = abs_diff / (np.abs(grad_numerical) + 1e-12)

            if verbose:
                print("Analytical Grad:\n", grad_analytical)
                print("Numerical Grad:\n", grad_numerical)
                print("Absolute Diff:\n", abs_diff)
                print("Relative Diff:\n", rel_diff)

            max_abs_error = np.max(abs_diff)
            max_rel_error = np.max(rel_diff)
            passed = np.allclose(grad_analytical, grad_numerical, atol=atol, rtol=rtol)

            print(f"\n✅ Gradient Check {'PASSED' if passed else 'FAILED'}")
            print(f"Max Absolute Error: {max_abs_error:.2e}")
            print(f"Max Relative Error: {max_rel_error:.2e}")

            return passed

        if stage == 1:
            if n_con == 0:
                constraints_list = [dict(type="ineq", fun=q_limits_constraint, jac=q_limits_constraint_grad)]
            else:
                constraints_list = [
                    dict(type="ineq", fun=q_limits_constraint, jac=q_limits_constraint_grad),
                    dict(type="eq", fun=contact_model_constraint, jac=contact_model_constraint_grad),
                    dict(type="ineq", fun=force_magnitude_constraint, jac=force_magnitude_constraint_grad),
                ]
        elif stage == 2:
            if n_con == 0:
                constraints_list = [
                    dict(type="ineq", fun=q_limits_constraint, jac=q_limits_constraint_grad),
                    dict(type="eq", fun=arm_doa_constraint, jac=arm_doa_constraint_grad),
                ]
            else:
                constraints_list = [
                    dict(type="eq", fun=contact_model_constraint, jac=contact_model_constraint_grad),
                    dict(type="ineq", fun=q_limits_constraint, jac=q_limits_constraint_grad),
                    dict(type="ineq", fun=friction_cone_constraint, jac=friction_cone_constraint_grad),
                    dict(type="eq", fun=force_magnitude_constraint, jac=force_magnitude_constraint_grad),
                    dict(type="eq", fun=arm_doa_constraint, jac=arm_doa_constraint_grad),
                ]

        bounds_dq = [(-q_step_max[i], q_step_max[i]) for i in range(q_step_max.shape[0])]
        bounds_cf = [(0, 100), (-50, 50), (-50, 50)] * n_con
        bounds = bounds_dq + bounds_cf
        x0 = np.concatenate([np.zeros((n_dof)), contact_force_all], axis=0)

        # if stage == 2:
        #     for i in range(100):
        #         x = np.random.randn(x0.shape[0])
        #         x[n_dof:] = np.clip(x[n_dof:], 0, 100)
        #         check_gradient(objective_fn=objective, x0=x, verbose=False)

        res = minimize(
            fun=objective,
            jac=jacobian,
            constraints=constraints_list,
            x0=x0,
            bounds=bounds,
            method="SLSQP",
            options={"ftol": 1e-6, "disp": b_print_opt_details, "maxiter": 200},
        )

        res_var = res.x.reshape(-1)
        dq_a = res_var[:n_dof]
        cf = res_var[n_hand_dof:]

        res = {}
        res["q_a"] = curr_q_a + dq_a
        res["dq_a"] = dq_a
        res["cf"] = cf
        return res

    def ctrl_opt4(
        self,
        stage,
        dt,
        curr_q_a,
        target_q_f,
        desired_sum_force,
        desired_obj_mass,
        last_dq_a,
        ho_contacts=None,
        grasp_matrix=None,
        b_contact=True,
        b_print_opt_details=False,
    ):
        # hyper-parameters
        mu = self.mu
        gravity = desired_obj_mass * np.array([0, 0, -9.81, 0, 0, 0]).reshape(-1, 1)

        # variables for coding convenience
        n_arm_dof = self.robot.arm.n_dof
        n_hand_dof = self.robot.hand.n_dof
        n_dof = n_arm_dof + n_hand_dof
        doa2dof_matrix = self.robot_adaptor.doa2dof_matrix
        n_con = len(ho_contacts)
        joint_limits_f = self.robot_adaptor.joint_limits_f
        q_step_max = np.asarray(self.robot.doa_max_vel) * dt

        if b_contact and n_con:
            # compute grasp matrix
            contact_G = self.compute_grasp_matrix(ho_contacts) if grasp_matrix is None else grasp_matrix

            # compute Ks and contact jacobian
            updated_contacts = self.Ks(q_a=curr_q_a, contacts=ho_contacts)
            Ks_all = []
            contact_jaco_all = []
            contact_force_all = []
            for i, contact in enumerate(updated_contacts):
                Ks_all.append(contact["Ks"])
                contact_jaco_all.append(contact["jaco"])
                contact_force_all.append(contact["contact_force"][:3])
            Ks_all = block_diag(*Ks_all)
            contact_jaco_all = np.concatenate(contact_jaco_all, axis=0)
            contact_force_all = np.concatenate(contact_force_all, axis=0)
            Ks_jaco = Ks_all @ contact_jaco_all
        else:
            contact_force_all = np.zeros((0))

        # compute target hand base pose
        hand_base_name = self.robot.hand.base_name
        self.robot_adaptor.compute_fk_f(target_q_f)
        target_hb_pose = self.robot_adaptor.get_frame_pose(frame_name=hand_base_name)
        target_hb_pos, target_hb_ori = isometry3dToPosOri(target_hb_pose)

        # weights
        w_hb_pose = np.diag([0, 0, 100.0, 10.0, 10.0, 10.0])
        w_q_hand = 1.0 * np.eye(n_hand_dof)
        w_dqa = 0.001 * np.eye(n_dof)
        w_ddqa = [0.00001] * n_arm_dof + [0.001] * n_hand_dof
        w_ddqa = np.diag(w_ddqa)
        w_cp = np.diag([0.0, 100, 100])
        w_cp = block_diag(*[w_cp for _ in range(n_con)])
        w_cf = np.diag([0.0, 0.0, 0.0])
        w_cf = block_diag(*[w_cf for _ in range(n_con)])
        w_wrench = 1.0 * np.diag([1.0, 1, 1, 1, 1, 1])

        def objective(x):
            dq_a = x[:n_dof].copy()
            cf = x[n_dof:].copy()  # d contact_force
            q_a = curr_q_a + dq_a
            q_f = doa2dof_matrix @ q_a
            _, q_hand = q_f[:n_arm_dof], q_f[n_arm_dof:]

            # cost for hand qpos
            target_q_hand = target_q_f[n_arm_dof:]
            self.err_q_hand = err_q_hand = (q_hand - target_q_hand).reshape(-1, 1)
            cost_q_hand = err_q_hand.T @ w_q_hand @ err_q_hand

            # cost for dqa
            self.err_dqa = err_dqa = (dq_a / dt).reshape(-1, 1)
            cost_dqa = err_dqa.T @ w_dqa @ err_dqa

            # cost for ddqa
            self.err_ddqa = err_ddqa = ((dq_a - last_dq_a) / dt**2).reshape(-1, 1)
            cost_ddqa = err_ddqa.T @ w_ddqa @ err_ddqa

            cost_wrench = 0
            cost_tan_motion = 0
            cost_tan_cf = 0
            cost_sum_cf = 0
            if b_contact and n_con > 0:
                if stage == 1:
                    # cost tangential motion (restrict the tangential motion of contacts)
                    dp = contact_jaco_all @ dq_a.reshape(-1, 1)
                    self.err_cp = err_cp = dp
                    cost_tan_motion = err_cp.T @ w_cp @ err_cp

                if stage == 2:
                    # cost wrench
                    wrench = contact_G @ cf.reshape(-1, 1) + gravity
                    # G = contact_G.reshape(6, -1, 3).transpose(1, 0, 2)
                    # fts = np.matmul(G, cf.reshape(-1, 3)[:, :, None]).reshape(-1, 6)
                    # sum_forces_mag = np.sum(np.linalg.norm(fts[:, :3], axis=1))
                    # sum_torques_mag = np.sum(np.linalg.norm(fts[:, 3:], axis=1))
                    # wrench[:3] /= sum_forces_mag + 1e-8
                    # wrench[3:] /= sum_torques_mag + 1e-8
                    self.wrench = wrench
                    cost_wrench = wrench.T @ w_wrench @ wrench

                    # cost tangential force
                    dcf = Ks_jaco @ dq_a.reshape(-1, 1)
                    pred_next_cf = contact_force_all.reshape(-1, 1) + dcf
                    self.err_cf = err_cf = cf.reshape(-1, 1) - pred_next_cf
                    cost_tan_cf = err_cf.T @ w_cf @ err_cf

            cost_hb_pose = 0
            if stage == 1:
                self.robot_adaptor.compute_fk_a(q_a)
                hb_pose = self.robot_adaptor.get_frame_pose(frame_name=hand_base_name)
                hb_pos, hb_quat = isometry3dToPosQuat(hb_pose)

                err_hb_pos = hb_pos - target_hb_pos
                hb_ori = sciR.from_quat(hb_quat)
                err_hb_ori = (hb_ori * target_hb_ori.inv()).as_rotvec()
                self.err_hb_pose = err_hb_pose = np.concatenate([err_hb_pos, err_hb_ori], axis=0).reshape(-1, 1)  # 6D
                cost_hb_pose = err_hb_pose.T @ w_hb_pose @ err_hb_pose

            total_cost = (
                cost_dqa
                + cost_ddqa
                + cost_q_hand
                + cost_wrench
                + cost_tan_motion
                + cost_tan_cf
                + cost_hb_pose
                + cost_sum_cf
            )
            return total_cost.item()

        def jacobian(x):
            dq_a = x[:n_dof].copy()
            q_a = curr_q_a + dq_a
            grad = np.zeros(x.shape[0])

            # grad of cost_dqa
            err_dqa = self.err_dqa
            grad_dqa = 2.0 / dt * w_dqa @ err_dqa
            grad[:n_dof] += grad_dqa.reshape(-1)

            # grad of cost_ddqa
            err_ddqa = self.err_ddqa
            grad_dqa = 2.0 / dt**2 * w_ddqa @ err_ddqa
            grad[:n_dof] += grad_dqa.reshape(-1)

            # grad of cost_q_hand
            J_qhand_dqa = doa2dof_matrix[n_arm_dof:, :]  # (n_hand_dof, n_dof)
            err_q_hand = self.err_q_hand
            grad_dqa = 2.0 * (J_qhand_dqa.T @ w_q_hand @ err_q_hand).reshape(-1)  # shape: (n_dof,)
            grad[:n_dof] += grad_dqa.reshape(-1)

            if b_contact and n_con > 0:
                if stage == 1:
                    # grad of cost_tan_motion
                    err_cp = self.err_cp
                    grad_dqa = 2.0 * (contact_jaco_all.T @ w_cp @ err_cp).reshape(-1)  # shape (n_dof,)
                    grad[:n_dof] += grad_dqa.reshape(-1)

                if stage == 2:
                    # grad of cost_wrench
                    wrench = self.wrench
                    grad_cf = 2 * (contact_G.T @ w_wrench @ wrench)  # shape (n, 1)
                    grad[n_dof:] += grad_cf.reshape(-1)
                    # # grad of cost_tan_cf
                    # err_cf = self.err_cf
                    # grad_dqa = -2 * Ks_jaco.T @ w_cf @ err_cf  # shape: (n_dof, 1)
                    # grad_cf = 2 * w_cf @ err_cf  # shape: (n_con * 3, 1)
                    # grad[:n_dof] += grad_dqa.reshape(-1)
                    # grad[n_dof:] += grad_cf.reshape(-1)

            if stage == 1:
                # grad of cost_hb_pose
                self.robot_adaptor.compute_jaco_a(q_a)
                hb_jaco = self.robot_adaptor.get_frame_jaco(frame_name=hand_base_name, type="space")
                err_hb_pose = self.err_hb_pose
                grad_dqa = 2.0 * hb_jaco.T @ w_hb_pose @ err_hb_pose
                grad[:n_dof] += grad_dqa.reshape(-1)

            return grad

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

        def wrench_constraint(x):
            cf = x[n_dof:].reshape(-1, 3)
            wrench = contact_G @ cf.reshape(-1, 1) + gravity
            constraint = wrench[2] - 0.1 * gravity[2]  #  >= 0
            return constraint.reshape(-1)

        def arm_doa_constraint(x):
            dq_a_arm = x[:n_arm_dof].copy()
            constraint = dq_a_arm - 0  # == 0
            return constraint.reshape(-1)

        def arm_doa_constraint_grad(x):
            grad = np.zeros((n_arm_dof, x.shape[0]))
            grad[:, :n_arm_dof] = np.eye(n_arm_dof)
            return grad

        def check_gradient(objective_fn, x0, epsilon=1e-6, verbose=True, atol=1e-5, rtol=1e-3):
            """
            Compare analytical vs numerical gradients of a given objective function at x0.

            Parameters:
                objective_fn (function): Function that returns cost and gradient: cost, grad = objective_fn(x)
                x0 (np.ndarray): Point at which to check the gradient
                epsilon (float): Step size for finite difference
                verbose (bool): Whether to print differences
                atol (float): Absolute tolerance for error
                rtol (float): Relative tolerance for error
            """
            cost, grad_analytical = objective_fn(x0)
            grad_numerical = np.zeros_like(x0)

            for i in range(len(x0)):
                x_forward = x0.copy()
                x_backward = x0.copy()
                x_forward[i] += epsilon
                x_backward[i] -= epsilon
                f_forward, _ = objective_fn(x_forward)
                f_backward, _ = objective_fn(x_backward)
                grad_numerical[i] = (f_forward - f_backward) / (2 * epsilon)

            # Compute difference
            abs_diff = np.abs(grad_analytical - grad_numerical)
            rel_diff = abs_diff / (np.abs(grad_numerical) + 1e-12)

            if verbose:
                print("Analytical Grad:\n", grad_analytical)
                print("Numerical Grad:\n", grad_numerical)
                print("Absolute Diff:\n", abs_diff)
                print("Relative Diff:\n", rel_diff)

            max_abs_error = np.max(abs_diff)
            max_rel_error = np.max(rel_diff)
            passed = np.allclose(grad_analytical, grad_numerical, atol=atol, rtol=rtol)

            print(f"\n✅ Gradient Check {'PASSED' if passed else 'FAILED'}")
            print(f"Max Absolute Error: {max_abs_error:.2e}")
            print(f"Max Relative Error: {max_rel_error:.2e}")

            return passed

        if stage == 1:
            if n_con == 0:
                constraints_list = [dict(type="ineq", fun=q_limits_constraint, jac=q_limits_constraint_grad)]
            else:
                constraints_list = [
                    dict(type="ineq", fun=q_limits_constraint, jac=q_limits_constraint_grad),
                    dict(type="eq", fun=contact_model_constraint, jac=contact_model_constraint_grad),
                    dict(type="ineq", fun=force_magnitude_constraint, jac=force_magnitude_constraint_grad),
                ]
        elif stage == 2:
            if n_con == 0:
                constraints_list = [
                    dict(type="ineq", fun=q_limits_constraint, jac=q_limits_constraint_grad),
                    dict(type="eq", fun=arm_doa_constraint, jac=arm_doa_constraint_grad),
                ]
            else:
                constraints_list = [
                    dict(type="eq", fun=contact_model_constraint, jac=contact_model_constraint_grad),
                    dict(type="ineq", fun=q_limits_constraint, jac=q_limits_constraint_grad),
                    dict(type="ineq", fun=friction_cone_constraint, jac=friction_cone_constraint_grad),
                    # dict(type="eq", fun=force_magnitude_constraint, jac=force_magnitude_constraint_grad),
                    dict(type="eq", fun=arm_doa_constraint, jac=arm_doa_constraint_grad),
                    dict(type="ineq", fun=wrench_constraint),
                ]

        bounds_dq = [(-q_step_max[i], q_step_max[i]) for i in range(q_step_max.shape[0])]
        bounds_cf = [(0, 100), (-50, 50), (-50, 50)] * n_con
        bounds = bounds_dq + bounds_cf
        x0 = np.concatenate([np.zeros((n_dof)), contact_force_all], axis=0)

        # if stage == 2:
        #     for i in range(100):
        #         x = np.random.randn(x0.shape[0])
        #         x[n_dof:] = np.clip(x[n_dof:], 0, 100)
        #         check_gradient(objective_fn=objective, x0=x, verbose=False)

        res = minimize(
            fun=objective,
            # jac=jacobian,
            constraints=constraints_list,
            x0=x0,
            bounds=bounds,
            method="SLSQP",
            options={"ftol": 1e-6, "disp": b_print_opt_details, "maxiter": 200},
        )

        res_var = res.x.reshape(-1)
        dq_a = res_var[:n_dof]
        cf = res_var[n_dof:]

        if stage == 2:
            print(f"cf: {cf.reshape(-1, 3)}")
            print(f"self.wrench: {self.wrench.reshape(-1)}")

        res = {}
        res["q_a"] = curr_q_a + dq_a
        res["dq_a"] = dq_a
        res["cf"] = cf
        return res


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
