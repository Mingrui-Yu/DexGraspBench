import numpy as np

# Dummy dimensions
n_dof = 12
n_contacts = 4
desired_sum_force = 10.0


def force_magnitude_constraint(x):
    """
    Input normal forces must be positive.
    """
    cf = x[n_dof:].reshape(-1, 3)
    constraint = np.sum(cf[:, 0]) - desired_sum_force  # == 0
    return np.array([constraint])  # shape (1,)


def force_magnitude_constraint_grad(x):
    """
    Gradient of the sum of normal forces minus desired force.
    Vectorized implementation.
    """
    n_vars = x.shape[0]
    grad = np.zeros((1, n_vars))  # shape: (1, len(x))
    idx = np.arange(n_contacts) * 3 + 0  # index of normal force in each contact
    grad[0, n_dof + idx] = 1.0
    return grad  # shape: (1, len(x))


def numerical_gradient(f, x, eps=1e-6):
    """
    Finite difference numerical gradient.
    """
    x = x.copy()
    grad = np.zeros((1, len(x)))
    fx = f(x)
    for i in range(len(x)):
        x[i] += eps
        fx_eps = f(x)
        x[i] -= eps
        grad[0, i] = (fx_eps - fx) / eps
    return grad


def test_force_magnitude_constraint_gradient():
    x = np.random.randn(n_dof + n_contacts * 3)

    # Forward constraint and gradients
    analytical_grad = force_magnitude_constraint_grad(x)
    numerical_grad_ = numerical_gradient(force_magnitude_constraint, x)

    # Print comparison
    print("Analytical Grad:\n", analytical_grad)
    print("Numerical Grad:\n", numerical_grad_)
    error = np.linalg.norm(analytical_grad - numerical_grad_)
    print("Gradient error (L2 norm):", error)

    # Check if close
    assert np.allclose(analytical_grad, numerical_grad_, atol=1e-5), "Gradient mismatch!"


# Run test
test_force_magnitude_constraint_gradient()
