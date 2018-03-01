"""Tests for flik.linesearch.interpolations."""

import numpy as np
from nose.tools import assert_raises

from flik.linesearch.interpolations import cubic_interpolation


def test_cubic_interpolation_input():
    """Test the checking of the arguments for the cubic_interpolation function."""
    # We need some prototype correct arguments for the function
    var = np.array([0.1, 2.0])

    def func(x):
        return x.dot(x)

    def grad(x):
        return 2 * x

    direction = np.array([-1.0, 0])

    # Check raises for parameters that aren't in the check_input function
    #   alphas should be floats
    interval_start = 0.2
    interval_end = 0.8

    alpha_1 = "a"
    alpha_2 = 0.7
    assert_raises(TypeError, cubic_interpolation, var, func, grad, direction, interval_start,
                  interval_end, alpha_1, alpha_2)

    alpha_1 = 0.43
    alpha_2 = (2.3, 0.2)
    assert_raises(TypeError, cubic_interpolation, var, func, grad, direction, interval_start,
                  interval_end, alpha_1, alpha_2)

    #   0 < interval_start < interval_end <= 1
    alpha_1 = 0.3
    alpha_2 = 0.6

    interval_start = -1.0
    interval_end = 0.8
    assert_raises(ValueError, cubic_interpolation, var, func, grad, direction, interval_start,
                  interval_end, alpha_1, alpha_2)

    interval_start = 0.5
    interval_end = 2.0
    assert_raises(ValueError, cubic_interpolation, var, func, grad, direction, interval_start,
                  interval_end, alpha_1, alpha_2)

    #   interval_start < alpha_1 < alpha_2 < interval_end
    interval_start = 0.2
    interval_end = 0.8

    alpha_1 = 0.1
    alpha_2 = 0.7
    assert_raises(ValueError, cubic_interpolation, var, func, grad, direction, interval_start,
                  interval_end, alpha_1, alpha_2)

    alpha_1 = 0.3
    alpha_2 = 0.9
    assert_raises(ValueError, cubic_interpolation, var, func, grad, direction, interval_start,
                  interval_end, alpha_1, alpha_2)


def test_cubic_interpolation():
    """Test the cubic interpolation algorithm with a manual calculation."""
    # Let's pick the following function: f(x) = x.x
    def func(x):
        return x.dot(x)

    def grad(x):
        return 2 * x

    # We pick an initial vector and a corresponding descent direction
    var = np.array([0.0, -1.0, 2.0])
    direction = np.array([2.0, 1.0, 0.0])

    # In the interval
    interval_start = 0.01
    interval_end = 1.0
    # with initial guesses for alpha_1 and alpha_2 being
    alpha_1 = 0.1
    alpha_2 = 0.9
    # a manual calculation shows that the return of the function should be
    assert np.abs(cubic_interpolation(var, func, grad, direction, interval_start, interval_end,
                                      alpha_1, alpha_2) - 0.2) < 1.0e-8
