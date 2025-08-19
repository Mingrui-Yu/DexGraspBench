import numpy as np
from mr_utils.utils_calc import sciR, isometry3dToPosQuat, 

b_contact = True
stage = 1


def objective(x):
    dq_a = x[:n_dof].copy()
    cf = x[n_dof:].copy()  # d contact_force
    q_a = curr_q_a + dq_a
    q_f = doa2dof_matrix @ q_a
    _, q_hand = q_f[:n_arm_dof], q_f[n_arm_dof:]

    # cost for hand qpos
    target_q_hand = target_q_f[n_arm_dof:]
    err_q_hand = (q_hand - target_q_hand).reshape(-1, 1)
    cost_q_hand = err_q_hand.T @ w_q_hand @ err_q_hand

    cost_wrench = 0
    cost_tan_motion = 0
    cost_tan_cf = 0
    if b_contact and n_con > 0:
        if stage == 1:
            # cost tangential motion (restrict the tangential motion of contacts)
            dp = contact_jaco_all @ dq_a.reshape(-1, 1)
            err_cp = dp
            cost_tan_motion = err_cp.T @ w_cp @ err_cp

        if stage == 2:
            # cost wrench
            wrench = contact_G @ cf.reshape(-1, 1)
            cost_wrench = wrench.T @ w_wrench @ wrench

            # cost tangential force
            dcf = Ks_jaco @ dq_a.reshape(-1, 1)
            pred_next_cf = contact_force_all.reshape(-1, 1) + dcf
            err_cf = cf.reshape(-1, 1) - pred_next_cf
            cost_tan_cf = err_cf.T @ w_cf @ err_cf

    cost_hb_pose = 0
    if stage == 1:
        self.robot_adaptor.compute_fk_a(q_a.detach().numpy())
        hb_pose = self.robot_adaptor.get_frame_pose(frame_name=hand_base_name)
        hb_pos, hb_quat = isometry3dToPosQuat(hb_pose)

        err_hb_pos = hb_pos - target_hb_pos
        hb_ori = sciR.from_quat(hb_quat)
        err_hb_ori = (hb_ori * target_hb_ori.inv()).as_rotvec()
        err_hb_pose = np.concatenate([err_hb_pos, err_hb_ori], axis=0).reshape(-1, 1)  # 6D
        cost_hb_pose = err_hb_pose.T @ w_hb_pose @ err_hb_pose

    total_cost = cost_q_hand + cost_wrench + cost_tan_motion + cost_tan_cf + cost_hb_pose

    # compute gradient
    grad = np.zeros(x.shape[0])

    # grad of cost_q_hand
    J_qhand_dqa = doa2dof_matrix[n_arm_dof:, :]  # (n_hand_dof, n_dof)
    grad_dqa = 2.0 * (J_qhand_dqa.T @ w_q_hand @ err_q_hand).reshape(-1)  # shape: (n_dof,)
    grad[:n_dof] += grad_dqa.reshape(-1)

    if b_contact and n_con > 0:
        if stage == 1:
            # grad of cost_tan_motion
            grad_dqa = 2.0 * (contact_jaco_all.T @ w_cp @ err_cp).reshape(-1)  # shape (n_dof,)
            grad[:n_dof] += grad_dqa.reshape(-1)

        if stage == 2:
            # grad of cost_wrench
            grad_cf = 2 * (contact_G.T @ w_wrench @ wrench)  # shape (n, 1)
            grad[n_dof:] += grad_cf.reshape(-1)

            # grad of cost_tan_cf
            grad_dqa = -2 * Ks_jaco.T @ w_cf @ err_cf  # shape: (n_dof, 1)
            grad_cf = 2 * w_cf @ err_cf  # shape: (n_con * 3, 1)
            grad[:n_dof] += grad_dqa.reshape(-1)
            grad[n_dof:] += grad_cf.reshape(-1)

    if stage == 1:
        # grad of cost_hb_pose
        grad_dqa = 2.0 * hb_jaco.T @ w_hb_pose @ err_hb_pose
        grad[:n_dof] += grad_dqa.reshape(-1)

    else:
        raise NotImplementedError()

    return total_cost.item(), grad





# ------------------------------
# Example: check_gradient(objective_fn, x0)
# ------------------------------

# You must define these variables before running:
#   - n_dof
#   - objective(x): your full function
#   - initial state x0: e.g., x0 = np.random.randn(n_dof + n_con * 3)

# Example (dummy shape, replace with your actual values)
n_dof = 24
n_con = 3
x0 = np.random.randn(n_dof + n_con * 3)

# Run gradient check
check_gradient(objective, x0)
