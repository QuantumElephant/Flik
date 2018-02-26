"""Test for flik.linesearch.initial_step_length."""

import numpy as np

from nose.tools import assert_raises
from flik.linesearch.initial_step_length import (initial_alpha_grad_based,
                                                 initial_alpha_default,
                                                 initial_alpha_func_based)


def test_initial_alpha_default():
    """Test for returning the default alpha value."""
    assert initial_alpha_default() == 1.0


def test_initial_alpha_grad_based():
    """Test for selecting the initial alpha value (gradient-based)."""
    # Set parameters

    def grad(current_point):
        """Gradient of function=current_point**2."""
        return 2*current_point

    current_point = np.array([0.5])
    previous_point = np.array([0.9])
    current_step = np.array([-0.1])
    previous_step = np.array([-0.06])
    previous_alpha = 0.8

    # Checking input quality

    grad_w = "This is not callable"
    assert_raises(TypeError, initial_alpha_grad_based, grad_w, current_point, previous_point,
                  current_step, previous_step, previous_alpha)
    current_point_w = "This is not a numpy array, float, or int"
    assert_raises(TypeError, initial_alpha_grad_based, grad, current_point_w, previous_point,
                  current_step, previous_step, previous_alpha)
    previous_point_w = "This is not a numpy array, float, or int"
    assert_raises(TypeError, initial_alpha_grad_based, grad, current_point, previous_point_w,
                  current_step, previous_step, previous_alpha)
    current_step_w = "This is not a numpy array, float, or int"
    assert_raises(TypeError, initial_alpha_grad_based, grad, current_point, previous_point,
                  current_step_w, previous_step, previous_alpha)
    previous_step_w = "This is not a numpy array, float, or int"
    assert_raises(TypeError, initial_alpha_grad_based, grad, current_point, previous_point,
                  current_step, previous_step_w, previous_alpha)
    current_point_w = np.array([[1., 2.], [3., 4.]])
    assert_raises(TypeError, initial_alpha_grad_based, grad, current_point_w, previous_point,
                  current_step, previous_step, previous_alpha)
    previous_point_w = np.array([[1., 2.], [3., 4.]])
    assert_raises(TypeError, initial_alpha_grad_based, grad, current_point, previous_point_w,
                  current_step, previous_step, previous_alpha)
    current_step_w = np.array([[1., 2.], [3., 4.]])
    assert_raises(TypeError, initial_alpha_grad_based, grad, current_point, previous_point,
                  current_step_w, previous_step, previous_alpha)
    previous_step_w = np.array([[1., 2.], [3., 4.]])
    assert_raises(TypeError, initial_alpha_grad_based, grad, current_point, previous_point,
                  current_step, previous_step_w, previous_alpha)
    current_point_w = np.array([10])
    assert_raises(TypeError, initial_alpha_grad_based, grad, current_point_w, previous_point,
                  current_step, previous_step, previous_alpha)
    previous_point_w = np.array([11])
    assert_raises(TypeError, initial_alpha_grad_based, grad, current_point, previous_point_w,
                  current_step, previous_step, previous_alpha)
    current_step_w = np.array([12])
    assert_raises(TypeError, initial_alpha_grad_based, grad, current_point, previous_point,
                  current_step_w, previous_step, previous_alpha)
    previous_step_w = np.array([13])
    assert_raises(TypeError, initial_alpha_grad_based, grad, current_point, previous_point,
                  current_step, previous_step_w, previous_alpha)
    previous_alpha_w = 2
    assert_raises(TypeError, initial_alpha_grad_based, grad, current_point, previous_point,
                  current_step, previous_step, previous_alpha_w)
    previous_alpha_w = 1.4142
    assert_raises(ValueError, initial_alpha_grad_based, grad, current_point, previous_point,
                  current_step, previous_step, previous_alpha_w)
    current_point_w = np.array([1., 0.])
    current_step_w = np.array([0., 1.])
    assert_raises(ValueError, initial_alpha_grad_based, grad, current_point_w, previous_point,
                  current_step_w, previous_step, previous_alpha)

    # Making current_point, previous_point, current_step, and previous_step float and int
    current_point_float = 0.5
    assert initial_alpha_grad_based(grad, current_point_float, previous_point,
                                    current_step, previous_step, previous_alpha)
    current_point_int = 1
    assert initial_alpha_grad_based(grad, current_point_int, previous_point,
                                    current_step, previous_step, previous_alpha)
    previous_point_float = 0.9
    assert initial_alpha_grad_based(grad, current_point, previous_point_float,
                                    current_step, previous_step, previous_alpha)
    previous_point_int = 1
    assert initial_alpha_grad_based(grad, current_point, previous_point_int,
                                    current_step, previous_step, previous_alpha)
    current_step_float = -0.1
    assert initial_alpha_grad_based(grad, current_point, previous_point,
                                    current_step_float, previous_step, previous_alpha)
    current_step_int = -1
    assert initial_alpha_grad_based(grad, current_point, previous_point,
                                    current_step_int, previous_step, previous_alpha)
    previous_step_float = -0.1
    assert initial_alpha_grad_based(grad, current_point, previous_point,
                                    current_step, previous_step_float, previous_alpha)
    previous_step_int = -1
    assert initial_alpha_grad_based(grad, current_point, previous_point,
                                    current_step, previous_step_int, previous_alpha)

    # Checking return value

    # Checking output type
    assert isinstance(initial_alpha_grad_based(grad, current_point, previous_point,
                                               current_step, previous_step, previous_alpha),
                      float)
    # Checking output range
    assert (0 < initial_alpha_grad_based(grad, current_point, previous_point, current_step,
                                         previous_step, previous_alpha) <= 1)


def test_initial_alpha_func_based():
    """Test for selecting the initial alpha value (function-based)."""
    # Set parameters

    def func(current_point):
        """Objective function."""
        return float(current_point**2)

    def grad(current_point):
        """Gradient of function=current_point**2."""
        return 2*current_point

    current_point = np.array([1.8])
    previous_point = np.array([1.9])
    current_step = np.array([-0.6])

    # Checking input quality

    func_w = "This is not callable"
    assert_raises(TypeError, initial_alpha_func_based, func_w, grad, current_point,
                  previous_point, current_step)
    grad_w = "This is not callable"
    assert_raises(TypeError, initial_alpha_func_based, func, grad_w, current_point,
                  previous_point, current_step)
    current_point_w = "This is not a numpy array, float, or int"
    assert_raises(TypeError, initial_alpha_func_based, func, grad, current_point_w,
                  previous_point, current_step)
    previous_point_w = "This is not a numpy array, float, or int"
    assert_raises(TypeError, initial_alpha_func_based, func, grad, current_point,
                  previous_point_w, current_step)
    current_step_w = "This is not a numpy array, float, or int"
    assert_raises(TypeError, initial_alpha_func_based, func, grad, current_point,
                  previous_point, current_step_w)
    current_point_w = np.array([[1., 2.], [3., 4.]])
    assert_raises(TypeError, initial_alpha_func_based, func, grad, current_point_w,
                  previous_point, current_step)
    previous_point_w = np.array([[1., 2.], [3., 4.]])
    assert_raises(TypeError, initial_alpha_func_based, func, grad, current_point,
                  previous_point_w, current_step)
    current_step_w = np.array([[1., 2.], [3., 4.]])
    assert_raises(TypeError, initial_alpha_func_based, func, grad, current_point,
                  previous_point, current_step_w)
    current_point_w = np.array([10])
    assert_raises(TypeError, initial_alpha_func_based, func, grad, current_point_w,
                  previous_point, current_step)
    previous_point_w = np.array([11])
    assert_raises(TypeError, initial_alpha_func_based, func, grad, current_point,
                  previous_point_w, current_step)
    current_step_w = np.array([12])
    assert_raises(TypeError, initial_alpha_func_based, func, grad, current_point,
                  previous_point, current_step_w)
    current_point_w = np.array([1., 0.])
    current_step_w = np.array([0., 1.])
    assert_raises(ValueError, initial_alpha_func_based, func, grad, current_point_w,
                  previous_point, current_step_w)

    # Making current_point, previous_point, current_step, and previous_step float and int
    current_point_float = 1.8
    assert initial_alpha_func_based(func, grad, current_point_float, previous_point, current_step)
    current_point_int = 2
    assert initial_alpha_func_based(func, grad, current_point_int, previous_point, current_step)
    previous_point_float = 1.9
    assert initial_alpha_func_based(func, grad, current_point, previous_point_float, current_step)
    previous_point_int = 2
    assert initial_alpha_func_based(func, grad, current_point, previous_point_int, current_step)
    current_step_float = -0.06
    assert initial_alpha_func_based(func, grad, current_point, previous_point, current_step_float)
    current_step_int = -1
    assert initial_alpha_func_based(func, grad, current_point, previous_point, current_step_int)

    # Checking return value

    # Checking output type
    assert isinstance(initial_alpha_func_based(func, grad, current_point,
                                               previous_point, current_step), float)

    # Checking output range
    assert 0 < initial_alpha_func_based(func, grad, current_point,
                                        previous_point, current_step) <= 1
