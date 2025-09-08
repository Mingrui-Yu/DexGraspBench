import numpy as np

# ====== toy data for testing ======
n_dof = 10
n_hand_dof = 5
n_con = 4

# 随机生成测试数据
np.random.seed(0)
q_f2a_direction = np.random.choice([-1.0, 1.0], size=n_dof)
contact_force_all = np.random.randn(3 * n_con)
contact_jaco_h = np.random.randn(3 * n_con, n_hand_dof)

# 测试输入
x = np.random.randn(n_dof + 3 * n_con)


def increase_joint_2_constraint(x):
    cf = x[n_dof:].copy()
    dcf = cf - contact_force_all

    idx_normal = np.arange(0, n_con * 3, 3)
    J_n = contact_jaco_h[idx_normal, :]  # (n_con, n_hand_dof)
    dtau_n = J_n.T @ dcf.reshape(-1, 3)[:, 0].reshape(-1, 1)

    constraint = np.diag(q_f2a_direction[-n_hand_dof:]) @ dtau_n
    return constraint.reshape(-1)  # >= 0


def increase_joint_2_constraint_grad(x):
    idx_normal = np.arange(0, n_con * 3, 3)
    J_n = contact_jaco_h[idx_normal, :]  # (n_con, n_hand_dof)

    # grad wrt dq_a: zero
    grad_dq_a = np.zeros((n_hand_dof, n_dof))

    # grad wrt cf
    grad_cf = np.zeros((n_hand_dof, 3 * n_con))
    grad_cf[:, idx_normal] = np.diag(q_f2a_direction[-n_hand_dof:]) @ J_n.T

    # full jacobian
    jacobian = np.hstack([grad_dq_a, grad_cf])  # shape: (n_hand_dof, n_dof + 3*n_con)
    return jacobian


# ====== finite difference check ======
def numerical_jacobian(func, x, eps=1e-6):
    f0 = func(x)
    jac = np.zeros((len(f0), len(x)))
    for i in range(len(x)):
        x_pos = x.copy()
        x_pos[i] += eps
        f_pos = func(x_pos)

        x_neg = x.copy()
        x_neg[i] -= eps
        f_neg = func(x_neg)

        jac[:, i] = (f_pos - f_neg) / (2 * eps)
    return jac


if __name__ == "__main__":
    analytic = increase_joint_2_constraint_grad(x)
    numeric = numerical_jacobian(increase_joint_2_constraint, x)

    diff = np.linalg.norm(analytic - numeric) / (np.linalg.norm(numeric) + 1e-12)

    print("Analytic Jacobian shape:", analytic.shape)
    print("Numeric Jacobian shape:", numeric.shape)
    print("Relative error:", diff)

    if diff < 1e-6:
        print("✅ Gradient check PASSED")
    else:
        print("❌ Gradient check FAILED")
