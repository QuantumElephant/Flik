"""Tests for flik.linesearch.conditions."""
import numpy as np
from numpy.testing import assert_raises
from flik.linesearch.conditions import strong_wolfe
from flik.linesearch.conditions import soft_wolfe


def test_strong_wolfe():
    """Test for the strong_wolfe function."""
    def grad(current_point):
        """Gradient of function :math:`current_point**2`."""
        return 2 * current_point

    # Set parameters
    current_point = np.array([1.8])
    current_step = np.array([-0.6])
    alpha = 0.8

    c_2_w = 1
    assert_raises(TypeError, strong_wolfe, grad, current_point, current_step, alpha, c_2_w)
    c_2_w = -0.209
    assert_raises(ValueError, strong_wolfe, grad, current_point, current_step, alpha, c_2_w)
    c_2_w = 0.9

    # Making current_point and current_step float and int
    current_point_float = 1.8
    assert strong_wolfe(grad, current_point_float, current_step, alpha, c_2_w)
    current_point_int = 2
    assert strong_wolfe(grad, current_point_int, current_step, alpha, c_2_w)
    current_step_float = -0.6
    assert strong_wolfe(grad, current_point, current_step_float, alpha, c_2_w)
    current_step_int = -1
    assert strong_wolfe(grad, current_point, current_step_int, alpha, c_2_w)

    # Checking condition
    assert strong_wolfe(grad, current_point, current_step, alpha, c_2_w)
    assert not strong_wolfe(grad, current_point, -current_step, alpha, c_2_w)


def test_soft_wolfe():
    """Test soft wolfe/curvature condition."""
    # Set up parameters
    def grad(val):
        """Right gradient of objective function. Returns 1-dim array."""
        return 2 * val

    direction = np.array([2., 1., 3.])
    alpha = 0.1
    val = np.array([2., 1., 3.])

    # Check input quality
    val_w = [1., 2., 3.]
    assert_raises(TypeError, soft_wolfe, grad, val_w, alpha, direction)
    grad_w = 2
    assert_raises(TypeError, soft_wolfe, grad_w, val, alpha, direction)
    alpha_w = 1
    assert_raises(TypeError, soft_wolfe, grad, val, alpha_w, direction)
    direction_w = [2., 1., 3.]
    assert_raises(TypeError, soft_wolfe, grad, val, alpha, direction_w)
    direction_w = np.array([2., 1.])
    assert_raises(ValueError, soft_wolfe, grad, val, alpha, direction_w)
    const2_w = 1.1
    assert_raises(ValueError, soft_wolfe, grad, val, alpha, direction, const2_w)

    def grad_ww(val):
        """Wrong gradient of objective function. Returns scalar."""
        return val.dot(val)

    assert_raises(TypeError, soft_wolfe, grad_ww, val, alpha, direction)

    # Check condition
    assert soft_wolfe(grad, val, alpha, direction)
    assert not soft_wolfe(grad, val, alpha, -direction)
