"""Conditions for the line search algorithm."""
import numpy as np
from flik.linesearch.tools import check_input


def wolfe(grad, point, step, alpha, const2=0.9, strong_wolfe=True):
    """Check if the given alpha satisfies the Wolfe condition.

    Parameters
    ----------
    grad: callable
        Gradient of the objective function.
    point: {np.ndarray(N,), float, int}
        Current iteration point.
        If a float or int is given, they are converted to a np.ndarray.
    step: {np.ndarray(N,), float, int}
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
        If point and step are not numpy arrays.
        If point and step are not 1-dimensional vectors.
        If the elements of point and step are not floats.
        If alpha and const2 are not floats.
    ValueError
        If point and step have different sizes.
        If alpha is not in the interval [0,1].
        If const2 is not in the interval (0,1).

    Returns
    -------
    bool
        True if the condition is satisfied, False otherwise

    """
    check_input(grad=grad, var=point, direction=step, alpha=alpha)
    if not isinstance(const2, float):
        raise TypeError("const2 should be a float")
    if not 0.0 < const2 < 1.0:
        raise ValueError("const2 should be in the interval (0, 1)")

    if isinstance(point, (int, float)):
        point = np.array(point, dtype=float, ndmin=1)
    if isinstance(step, (int, float)):
        step = np.array(step, dtype=float, ndmin=1)

    left = step.dot(grad(point + alpha * step))
    right = const2 * step.dot(grad(point))

    if strong_wolfe:
        return abs(left) <= abs(right)
    else:
        return left >= right


def armijo(func, grad, point, step, alpha, const1=1e-4):
    r"""Check if given alpha satisfies the Armijo condition.

    Armijo condition is satisfied if
    :math:`f(x_k + \alpha s_k) - f(x_k) \leq c_1 \alpha \nabla f(x_k)^T s_k`

    Parameters
    ----------
    func: callable
        Objective function.
    grad: callable
        Function to evaluate gradient of the function.
    point: {np.ndarray(N,), float, int}
        Value of variables at k-iteration.
    step: {np.ndarray(N,), float, int}
        Descent direction.
    alpha: float
        Step length
    const1: {float, 1e-4}
        Condition parameter.

    Returns
    -------
    bool
        True if the condition is satisfied, False otherwise.

    """
    check_input(func=func, grad=grad, var=point, direction=step,
                alpha=alpha)
    if not isinstance(const1, float):
        raise TypeError("const1 should be float")
    if not 0.0 < const1 < 1.0:
        raise ValueError("const1 should have a value between 0. and 1.")

    # Evaluate the function and gradient
    func_current = func(point)
    grad_current = grad(point)
    func_step = func(point + alpha*step)

    # Check the condition
    return func_current - func_step <= const1 * alpha * step.dot(grad_current)
