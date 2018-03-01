"""Conditions for the line search algorithm."""
import numpy as np
from flik.linesearch.tools import check_input


def strong_wolfe(grad, current_point, current_step, alpha, c_2=0.9):
    """Check check that the given alpha satisfies the strong Wolfe condition.

    Parameters
    ----------
    grad: callable
        Gradient of the objective function.
    current_point: {np.ndarray, float, int}
        Current iteration point.
        If a float or int is given, they are converted to a np.ndarray.
    current_step: {np.ndarray, float, int}
        Current direction along which the minimum will be searched.
        If a float or int is given, they are converted to a np.ndarray.
    alpha: float
        Step length.
    c_2: float
        Strong Wolfe condition parameter.

    Raises
    -------
    TypeError
        If grad is not a callable object.
        If current_point and current_step are not numpy arrays.
        If current_point and current_step are not 1-dimensional vectors.
        If the elements of current_point and current_step are not floats.
        If alpha and c_2 are not floats.
    ValueError
        If current_point and current_step have different sizes.
        If alpha is not in the interval (0,1].
        If c_2 is not in the interval (0,1).

    Returns
    -------
    bool
        True if the condition is satisfied, False otherwise

    """
    check_input(grad=grad, var=current_point, direction=current_step, alpha=alpha)
    if not isinstance(c_2, float):
        raise TypeError("c_2 should be a float")
    if not 0.0 < c_2 < 1.0:
        raise ValueError("c_2 should be in the interval (0,1)")

    if isinstance(current_point, (int, float)):
        current_point = np.array(current_point, dtype=float, ndmin=1)
    if isinstance(current_step, (int, float)):
        current_step = np.array(current_step, dtype=float, ndmin=1)

    current_grad = grad(current_point)
    next_grad = grad(current_point + alpha*current_step)
    return abs(current_step.dot(next_grad)) <= c_2*abs(current_step.dot(current_grad))
