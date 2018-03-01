"""Tests for flik.linesearch.conditions."""

import numpy as np
from nose.tools import assert_raises
from flik.linesearch.conditions import strong_wolfe


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
