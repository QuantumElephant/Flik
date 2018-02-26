"""Selection of the initial step length."""

import numpy as np


def initial_alpha_default():
    """Return default value for the initial alpha.

    Returns
    -------
    float:
        Default value for alpha_0 = 1.0

    """
    return 1.0


def initial_alpha_grad_based(grad, current_point, previous_point, current_step, previous_step,
                             previous_alpha=1.0):
    """Select initial alpha value based on the value of the gradient on the previous point.

    Parameters
    ----------
    grad: callable
        Gradient of the objective function
    current_point: np.ndarray, float, int
        Current iteration point
        If a float or int are given they are converted to a numpy array
    previous_point: np.ndarray, float, int
        Previous iteration point
        If a float or int are given they are converted to a numpy array
    current_step: np.ndarray, float, int
        Current direction along which the minimum will be searched
        If a float or int are given they are converted to a numpy array
    previous_step: np.ndarray, float, int
        Previous direction along which the minimum was searched
        If a float or int are given they are converted to a numpy array
    previous_alpha: float
        Step length of the previous iteration

    Raises
    -------
    TypeError
        If grad is not a callable object
        If current_point and previous_point are not np.ndarrays
        If current_step and previous_step are not np.ndarrays
        If current_point and previous_point are not 1-dimensional vectors
        If current_step and previous_step are not 1-dimensional vectors
        If the elements of current_point and previous_point are not floats
        If the elements of current_step and previous_step are not floats
        If previous_alpha is not a float
    ValueError
        If previous_alpha is not in the interval (0,1]
        If the gradient in the current point is orthogonal to current_step

    Returns
    -------
    float:
        Estimate of the initial step length

    """
    if not callable(grad):
        raise TypeError("The gradient should be a function")

    if isinstance(current_point, (float, int)):
        current_point = np.array([current_point], dtype=float)
    elif not isinstance(current_point, np.ndarray):
        raise TypeError("Current point should be a numpy array, float, or int")

    if isinstance(previous_point, (float, int)):
        previous_point = np.array([previous_point], dtype=float)
    elif not isinstance(previous_point, np.ndarray):
        raise TypeError("Previous point should be a numpy array, float, or int")

    if isinstance(current_step, (float, int)):
        current_step = np.array([current_step], dtype=float)
    elif not isinstance(current_step, np.ndarray):
        raise TypeError("Current step should be a numpy array, float, or int")

    if isinstance(previous_step, (float, int)):
        previous_step = np.array([previous_step], dtype=float)
    elif not isinstance(previous_step, np.ndarray):
        raise TypeError("Previous step should be a numpy array, float, or int")

    if current_point.ndim != 1:
        raise TypeError("Current point should be given as a 1-dimensional vector")

    if previous_point.ndim != 1:
        raise TypeError("Previous point should be given as a 1-dimensional vector")

    if current_step.ndim != 1:
        raise TypeError("Current step should be given as a 1-dimensional vector")

    if previous_step.ndim != 1:
        raise TypeError("Previous step should be given as a 1-dimensional vector")

    if current_point.dtype != float:
        raise TypeError("Current point should be given as a numpy array of floats")

    if previous_point.dtype != float:
        raise TypeError("Previous point should be given as a numpy array of floats")

    if current_step.dtype != float:
        raise TypeError("Current step should be given as a numpy array of floats")

    if previous_step.dtype != float:
        raise TypeError("Previous step should be given as a numpy array of floats")

    if not isinstance(previous_alpha, float):
        raise TypeError("Previous alpha should be a float")

    if not 0 < previous_alpha <= 1:
        raise ValueError("Previous alpha should be in the interval (0,1]")

    if grad(current_point).dot(current_step) == 0:
        raise ValueError("The gradient and the direction search should not be orthogonal")

    return (previous_alpha*previous_step.dot(grad(previous_point)) /
            (current_step.dot(grad(current_point))))


def initial_alpha_func_based(func, grad, current_point, previous_point, current_step):
    """Select initial alpha value based on the value of the function on the previous point.

    Parameters
    ----------
    func: callable
        Objective function
    grad: callable
        Gradient of the objective function
    current_point: np.ndarray, float, int
        Current iteration point
        If a float or int are given they are converted to a numpy array
    previous_point: np.ndarray, float, int
        Previous iteration point
        If a float or int are given they are converted to a numpy array
    current_step: np.ndarray, float, int
        Current direction along which the minimum will be searched
        If a float or int are given they are converted to a numpy array

    Raises
    -------
    TypeError
        If func and grad are not callable objects
        If current_point and previous_point are not np.ndarrays
        If current_step is not a np.ndarray
        If current_point and previous_point are not 1-dimensional vectors
        If current_step is not a 1-dimensional vector
        If the elements of current_point and previous_point are not floats
        If the elements of current_step are not floats
    ValueError
        If the gradient in the current point is orthogonal to current_step

    Returns
    -------
    float:
        Estimate of the initial step length

    """
    if not callable(func):
        raise TypeError("The objective function should be a function")

    if not callable(grad):
        raise TypeError("The gradient should be a function")

    if isinstance(current_point, (float, int)):
        current_point = np.array([current_point], dtype=float)
    elif not isinstance(current_point, np.ndarray):
        raise TypeError("Current point should be a numpy array, float, or int")

    if isinstance(previous_point, (float, int)):
        previous_point = np.array([previous_point], dtype=float)
    elif not isinstance(previous_point, np.ndarray):
        raise TypeError("Previous point should be a numpy array, float, or int")

    if isinstance(current_step, (float, int)):
        current_step = np.array([current_step], dtype=float)
    elif not isinstance(current_step, np.ndarray):
        raise TypeError("Current step should be a numpy array, float, or int")

    if current_point.ndim != 1:
        raise TypeError("Current point should be given as a 1-dimensional vector")

    if previous_point.ndim != 1:
        raise TypeError("Previous point should be given as a 1-dimensional vector")

    if current_step.ndim != 1:
        raise TypeError("Current step should be given as a 1-dimensional vector")

    if current_point.dtype != float:
        raise TypeError("Current point should be given as a numpy array of floats")

    if previous_point.dtype != float:
        raise TypeError("Previous point should be given as a numpy array of floats")

    if current_step.dtype != float:
        raise TypeError("Current step should be given as a numpy array of floats")

    if grad(current_point).dot(current_step) == 0:
        raise ValueError("The gradient and the direction search should not be orthogonal")

    return 2.0*(func(current_point)-func(previous_point)) / \
               (current_step.dot(grad(current_point)))
