"""Tests for flik.linesearch.conditions."""

import numpy as np

from nose.tools import assert_raises
from flik.linesearch.conditions import strong_wolfe


def test_strong_wolfe():
    """Test for the strong_wolfe function."""
    # Set parameters

    def grad(current_point):
        """Gradient of function=current_point**2."""
        return 2 * current_point

    current_point = np.array([1.8])
    current_step = np.array([-0.6])
    alpha = 0.8

    # Checking input quality

    grad_w = "This is not callable"
    assert_raises(TypeError, strong_wolfe, grad_w, current_point, current_step, alpha)
    current_point_w = "This is not a numpy array, float, or int"
    assert_raises(TypeError, strong_wolfe, grad, current_point_w, current_step, alpha)
    current_step_w = "This is not a numpy array, float, or int"
    assert_raises(TypeError, strong_wolfe, grad, current_point, current_step_w, alpha)
    current_point_w = np.array([[1., 2.], [3., 4.]])
    assert_raises(TypeError, strong_wolfe, grad, current_point_w, current_step, alpha)
    current_step_w = np.array([[1., 2.], [3., 4.]])
    assert_raises(TypeError, strong_wolfe, grad, current_point, current_step_w, alpha)
    current_point_w = np.array([1])
    assert_raises(TypeError, strong_wolfe, grad, current_point_w, current_step, alpha)
    current_step_w = np.array([2])
    assert_raises(TypeError, strong_wolfe, grad, current_point, current_step_w, alpha)
    alpha_w = 2
    assert_raises(TypeError, strong_wolfe, grad, current_point, current_step, alpha_w)
    c_2_w = 1
    assert_raises(TypeError, strong_wolfe, grad, current_point, current_step, alpha, c_2_w)
    current_point_w = np.array([1., 2.])
    current_step_w = np.array([0.9])
    assert_raises(ValueError, strong_wolfe, grad, current_point_w, current_step_w, alpha)
    alpha_w = 1.4142
    assert_raises(ValueError, strong_wolfe, grad, current_point, current_step, alpha_w)
    c_2_w = -0.209
    assert_raises(ValueError, strong_wolfe, grad, current_point, current_step, alpha, c_2_w)

    # Making current_point and current_step float and int
    current_point_float = 1.8
    assert strong_wolfe(grad, current_point_float, current_step, alpha)
    current_point_int = 2
    assert strong_wolfe(grad, current_point_int, current_step, alpha)
    current_step_float = -0.6
    assert strong_wolfe(grad, current_point, current_step_float, alpha)
    current_step_int = -1
    assert strong_wolfe(grad, current_point, current_step_int, alpha)

    # Checking condition
    assert strong_wolfe(grad, current_point, current_step, alpha)
    assert not strong_wolfe(grad, current_point, -current_step, alpha)
