"""Conditions for the line search algorithm."""
import numpy as np
from flik.linesearch.tools import check_input


def wolfe(grad, current_point, current_step, alpha, const2=0.9, strong_wolfe=True):
    """Check if the given alpha satisfies the Wolfe condition.

    Parameters
    ----------
    grad: callable
        Gradient of the objective function.
    current_point: {np.ndarray(N,), float, int}
        Current iteration point.
        If a float or int is given, they are converted to a np.ndarray.
    current_step: {np.ndarray(N,), float, int}
        Descent direction.
        If a float or int is given, they are converted to a np.ndarray.
    alpha: float
        Step length.
    const2: {float, 0.9}
        Condition parameter.
    strong_wolfe: {bool, False}
        If False, then checks for soft Wolfe condition.
        Default is strong Wolfe condition.

    Raises
    -------
    TypeError
        If grad is not a callable object.
        If current_point and current_step are not numpy arrays.
        If current_point and current_step are not 1-dimensional vectors.
        If the elements of current_point and current_step are not floats.
        If alpha and const2 are not floats.
    ValueError
        If current_point and current_step have different sizes.
        If alpha is not in the interval [0,1].
        If const2 is not in the interval (0,1).

    Returns
    -------
    bool
        True if the condition is satisfied, False otherwise

    """
    check_input(grad=grad, var=current_point, direction=current_step, alpha=alpha)
    if not isinstance(const2, float):
        raise TypeError("const2 should be a float")
    if not 0.0 < const2 < 1.0:
        raise ValueError("const2 should be in the interval (0, 1)")

    if isinstance(current_point, (int, float)):
        current_point = np.array(current_point, dtype=float, ndmin=1)
    if isinstance(current_step, (int, float)):
        current_step = np.array(current_step, dtype=float, ndmin=1)

    left = current_step.dot(grad(current_point + alpha * current_step))
    right = const2 * current_step.dot(grad(current_point))

    if strong_wolfe:
        return abs(left) <= abs(right)
    else:
        return left >= right


def armijo(var=None, func=None, grad=None, direction=None, alpha=None, const1=1e-4):
    r"""Check if Armijo condition is satisfied for certain alpha.

    A sufficient decrease is measured by:
    :math:`f(x_k + \alpha s_k) - f(x_k) \leq c_1 \alpha \nabla f(x_k)^T s_k`

    Parameters:
    -----------
    func: callable
        Objective function
    grad: callable
        Function to evaluate Gradient of the function
    vec: np.ndarray((N,)), for N variables
        Value of variables at k-iteration
    desc: np.ndarray((N,)) for N variables
        Descent direction
    alpha: float
        Step length
    const1: float
        Condition constant
    Returns:
    --------
    bool
        True if the condition is satisfied, False otherwise
    """
    check_input(func=func, grad=grad, var=var, direction=direction, alpha=alpha)
    if not isinstance(const1, float):
        raise TypeError("alpha should have a value between 0. and 1.")
    if not 0.0 < const1 < 1.0:
        raise ValueError("const1 should have a value between 0. and 1.")

    # Evaluate the function and gradient
    funck = func(var)
    gradk = grad(var)
    fstep = func(var + alpha*direction)

    # Check the condition
    return fstep - funck <= const1*alpha*direction.dot(gradk)
