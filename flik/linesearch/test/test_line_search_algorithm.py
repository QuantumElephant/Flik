"""Test for flik.linesearch.line_search_algorithm."""

import numpy as np

from nose.tools import assert_raises
from flik.linesearch.interpolations import cubic_interpolation
from flik.linesearch.zoom import zoom
from flik.linesearch.line_search_algorithm import line_search_strong_wolfe


def test_initial_alpha_func_based():
    """Test for selecting the initial alpha value (function-based)."""
    # Set parameters

    def func(current_point):
        """Objective function."""
        return float(current_point**2)

    def grad(current_point):
        """Gradient of function :math:`current_point**2`."""
        return 2*current_point

    current_point = np.array([1.8])
    current_step = np.array([-0.6])
    alpha_max = 1.0
    c_1 = 1e-4
    c_2 = 0.9

    # Checking input quality

    c_1_w = 209
    assert_raises(TypeError, line_search_strong_wolfe, func, grad,
                  current_point, current_step, alpha_max, c_1_w, c_2)
    c_2_w = 704
    assert_raises(TypeError, line_search_strong_wolfe, func, grad,
                  current_point, current_step, alpha_max, c_1, c_2_w)
    c_1_w = 2.09
    assert_raises(ValueError, line_search_strong_wolfe, func, grad,
                  current_point, current_step, alpha_max, c_1_w, c_2)
    c_2_w = 7.04
    assert_raises(ValueError, line_search_strong_wolfe, func, grad,
                  current_point, current_step, alpha_max, c_1, c_2_w)

    # Making current_point and current_step float and int
    current_point_float = 1.8
    assert line_search_strong_wolfe(func, grad, current_point_float,
                                    current_step, alpha_max, c_1, c_2)
    current_point_int = 2
    assert line_search_strong_wolfe(func, grad, current_point_int,
                                    current_step, alpha_max, c_1, c_2)
    current_step_float = 1.8
    assert line_search_strong_wolfe(func, grad, current_point,
                                    current_step_float, alpha_max, c_1, c_2)
    current_step_int = 2
    assert line_search_strong_wolfe(func, grad, current_point,
                                    current_step_int, alpha_max, c_1, c_2)

    # Checking return value

    # Checking output type
    assert isinstance(line_search_strong_wolfe(func, grad, current_point,
                                               current_step, alpha_max,
                                               c_1, c_2), float)

    # Checking output range
    assert 0 < line_search_strong_wolfe(func, grad, current_point,
                                        current_step, alpha_max,
                                        c_1, c_2) <= 1
