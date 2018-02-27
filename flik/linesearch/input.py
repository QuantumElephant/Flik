""" Input processing of Line Search algorithm."""

import numpy as np

def check_input(var, func, grad, direction, alpha):
    r"""Input control in Line Search algorithm

    Parameters:
    -----------
    var: np.array((N,), dtype=float)
        Variables' current values
    func: callable
        Function :math:`f(x)`
    grad: callable
        Gradient of the function :math:`\nabla f(x)`
    direction: np.ndarray((N,), dtype=float) for N variables
        Direction vector for next step
    alpha: float
        Step length

    Raises:
    -------
    TypeError:
        If var is not a float or a 1-D np.ndarray
        If func is not a callable
        If grad is not a callable
        If direction is not a float or a 1-D np.ndarray
        If alpha is not a float
    ValueError:
        If var and direction don't have the same shape
        If alpha is outside the interval (0., 1.]
    """
    if isinstance(var, (int,float)):
        var = np.array(var, dtype=float, ndim=1)
    elif not (isinstance(var, np.ndarray) and var.ndim == 1):
        raise TypeError("Variable vector should be a float or a 1-D numpy.ndarray")
    if not callable(func):
        raise TypeError("func must be callable")
    if not callable(grad):
        raise TypeError("grad must be callable")
    if isinstance(direction, (int, float)):
        direction = np.array(direction, dtype=float, ndim=1)
    elif not (isinstance(direction, np.ndarray) and direction.ndim == 1):
        raise TypeError("The direction vector should be provided as a float or"
                        " a numpy.ndarray")
    if not isinstance(alpha, float):
        raise TypeError("Alpha should be provided as a float")
    if var.shape != direction.shape:
        raise ValueError("The shape of vec and direction should be the same")
    if not  0. < alpha <= 1.0:
        raise ValueError("Alpha's value should be in the interval (0., 1.]")
