"""Line search algorithm."""

import numpy as np

from flik.linesearch.tools import check_input
from flik.linesearch.interpolations import cubic_interpolation
from flik.linesearch.zoom import zoom


def line_search_strong_wolfe(func, grad, current_point, current_step,
                             alpha_max=1.0, c_1=1e-4, c_2=0.9):
    """Algorithm to obtain an alpha that satisfies the strong Wolfe condition.

    Parameters
    ----------
    func: callable
        Objective function.
    grad: callable
        Gradient of the objective function.
    current_point: {np.ndarray, float, int}
        Current iteration point.
        If a float or int are given they are converted to a numpy array.
    current_step: {np.ndarray, float, int}
        Current direction along which the minimum will be searched.
        If a float or int are given they are converted to a numpy array.
    alpha_max: float
        Maximum allowed alpha.
    c_1: float
        Armijo condition parameter.
    c_2: float
        Strong Wolfe condition parameter.

    Raises
    ------
    TypeError
        If func and grad are not callable objects.
        If current_point and current_step are not np.arrays.
        If the elements of current_point and current_step are not floats.
        If alpha_max, c_1, and c_2 are not floats.
    ValueError
        If current_point and current_step have different sizes.
        If alpha_max is not in the interval (0,1].
        If c_1 is not in the interval (0,c_2).
        If c_2 is not in the interval (c_1,1).

    Returns
    -------
    float
        Alpha value that satisfies the strong Wolfe condition.

    """
    check_input(func=func, grad=grad, var=current_point, direction=current_step,
                alpha=alpha_max)
    if isinstance(current_point, (int, float)):
        current_point = np.array(current_point, dtype=float, ndmin=1)

    if isinstance(current_step, (int, float)):
        current_point = np.array(current_step, dtype=float, ndmin=1)

    if not isinstance(c_1, float):
        raise TypeError("c_1 should be a float")

    if not isinstance(c_2, float):
        raise TypeError("c_2 should be a float")

    if not 0.0 < c_1 < c_2:
        raise ValueError("c_1 should be in the interval (0,c_2)")

    if not c_1 < c_2 < 1:
        raise ValueError("c_2 should be in the interval (c_1,1)")

    # Helper functions
    def phi(alpha):
        """Objective function of the line search."""
        return func(current_point + alpha*current_step)

    def deriv_phi(alpha):
        """Derivative of the objective function of the line search."""
        return current_step.dot(grad(current_point + alpha*current_step))

    previous_alpha = 0.0
    current_alpha = cubic_interpolation(func, grad, previous_alpha,
                                        alpha_max, current_point, current_step,
                                        previous_alpha, alpha_max)
    deriv_phi_0 = deriv_phi(0.0)

    while True:
        if phi(current_alpha) > phi(0.0) + c_1*current_alpha*deriv_phi_0:
            return zoom(current_point, func, grad, current_step,
                        previous_alpha, current_alpha, c_1, c_2)
        deriv_phi_alpha = deriv_phi(current_alpha)
        if abs(deriv_phi_alpha) <= - c_2*deriv_phi_0:
            return current_alpha
        elif deriv_phi_alpha >= 0.0:
            return zoom(current_point, func, grad, current_step,
                        current_alpha, previous_alpha, c_1, c_2)
        previous_alpha = current_alpha
        current_alpha = cubic_interpolation(func, grad, current_alpha,
                                            alpha_max, current_point, current_step,
                                            current_alpha, alpha_max)
