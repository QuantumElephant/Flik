"""Input processing of line search module."""
import itertools as it
import numpy as np


def check_input(*, var=None, direction=None, alpha=None,
                func=None, grad=None, func_val=None, grad_val=None):
    r"""Test input used in line search module.

    Parameters
    ----------
    var: {int, float, np.array((N,)), None}
        Variables' current values.
    func: {callable, None}
        Function :math:`f(x)`.
    grad: {callable, None}
        Gradient of the function :math:`\nabla f(x)`.
    func_val: {float, None}
        Value of the function evaluated at var.
    grad_val: {int, float, np.array((N,)), None}
        Gradient evaluated at var.
    direction: {int, float, np.ndarray((N,))}
        Direction vector for next step.
    alpha: {float, None}
        Step length.

    Raises
    ------
    TypeError
        If var is not an int, float, or one dimensional numpy array.
        If func is not a callable.
        If grad is not a callable.
        If func_val is not an int or float.
        If grad_val is not an int, float, or one dimensional numpy array.
        If direction is not an int, float, or one dimensional numpy array.
        If alpha is not an int or float.
    ValueError
        If alpha is not in the interval (0, 1].
        If var and direction do not have the same shape.
        If var and grad_val do not have the same shape.
        If direction and grad_val do not have the same shape.

    """
    if var is None:
        pass
    elif isinstance(var, (int, float)):
        var = np.array(var, dtype=float, ndmin=1)
    elif not (isinstance(var, np.ndarray) and var.ndim == 1):
        raise TypeError("Variable vector should be a float or a 1-D numpy.ndarray")

    if func is None:
        pass
    elif not callable(func):
        raise TypeError("func must be callable")

    if grad is None:
        pass
    elif not callable(grad):
        raise TypeError("grad must be callable")

    if func_val is None:
        pass
    elif not isinstance(func_val, (int, float)):
        raise TypeError('func_val must be a float.')

    if grad_val is None:
        pass
    elif isinstance(grad_val, (int, float)):
        grad_val = np.array(grad_val, dtype=float, ndmin=1)
    elif not (isinstance(grad_val, np.ndarray) and grad_val.ndim == 1):
        raise TypeError('grad_val must be a one dimensional numpy array.')

    if direction is None:
        pass
    elif isinstance(direction, (int, float)):
        direction = np.array(direction, dtype=float, ndmin=1)
    elif not (isinstance(direction, np.ndarray) and direction.ndim == 1):
        raise TypeError("The direction vector should be provided as a float or a numpy array")

    if alpha is None:
        pass
    elif not isinstance(alpha, (int, float)):
        raise TypeError("Alpha should be provided as a float")
    elif not 0 <= alpha <= 1:
        raise ValueError("Alpha value should be in the interval [0., 1.]")

    name_dict = {'var': var, 'direction': direction, 'grad_val': grad_val}
    for name1, name2 in it.combinations(name_dict, 2):
        array1 = name_dict[name1]
        array2 = name_dict[name2]

        if array1 is None or array2 is None:
            continue
        if array1.shape != array2.shape:
            raise ValueError(f'{name1} and {name2} must have the same shape.')
