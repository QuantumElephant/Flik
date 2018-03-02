"""Tests for flik.linesearch.interpolations."""

import numpy as np
from nose.tools import assert_raises
from flik.linesearch.interpolations import cubic_interpolation, bs1, bs2


def test_cubic_interpolation():
    """Test the for the cubic_interpolation function with bijection."""
    def func(x):
        r"""Return value of function :math:`f(x)=x^T x`."""
        return x.dot(x)

    def grad(x):
        r"""Return gradient of function :math:`\nabla f(x)=2x`."""
        return 2 * x

    var = np.array([0.0, -1.0, 2.0])
    direction = np.array([2.0, 1.0, 0.0])
    interval_start = 0.01
    interval_end = 1.0
    alpha_1 = 0.1
    alpha_2 = 0.9

    # Check raises for parameters that aren't in the check_input function
    assert_raises(TypeError, cubic_interpolation, var, func, grad, direction, 0, 0.9, '0', 1)
    assert_raises(TypeError, cubic_interpolation, var, func, grad, direction, 0, 0.9, 0, '1')
    assert_raises(ValueError, cubic_interpolation, var, func, grad, direction, 0, 0.9, 0.1, 0.9)
    assert_raises(ValueError, cubic_interpolation, var, func, grad, direction, 0.1, 1, 0.1, 0.9)
    assert_raises(ValueError, cubic_interpolation, var, func, grad, direction, 1, 0.9, 0.1, 0.9)
    assert_raises(ValueError, cubic_interpolation, var, func, grad, direction, 0.1, 0, 0.1, 0.9)
    assert_raises(ValueError, cubic_interpolation, var, func, grad, direction, 0, 1, 0, 2)
    assert_raises(ValueError, cubic_interpolation, var, func, grad, direction, 0, 1, -1, 1)
    assert_raises(ValueError, cubic_interpolation, var, func, grad, direction, 0.1, 0.9, 1, 0)

    # a manual calculation shows that the return of the function should be
    assert np.abs(cubic_interpolation(var, func, grad, direction, alpha_1, alpha_2,
                                      interval_start, interval_end) - 0.2) < 1e-8

def test_bs1():
    """Test random-generating bisection method."""
    # Set parameters

    alpha_1 = 0.1
    alpha_2 = 0.9

    # Checking input quality

    alpha_1_w = 1
    assert_raises(TypeError, bs1, alpha_1_w, alpha_2)
    alpha_2_w = 2
    assert_raises(TypeError, bs1, alpha_1, alpha_2_w)
    alpha_1_w = 1.5
    assert_raises(ValueError, bs1, alpha_1_w, alpha_2)
    alpha_2_w = 2.5
    assert_raises(ValueError, bs1, alpha_1, alpha_2_w)

    # Checking output
    assert 0.0 <= bs1(alpha_1, alpha_2) <= 1.0

def test_bs2():
    """Test for simple (average) bisection method."""
    # Set parameters

    alpha_1 = 0.9
    alpha_2 = 0.1

    # Checking input quality

    alpha_1_w = 1
    assert_raises(TypeError, bs2, alpha_1_w, alpha_2)
    alpha_2_w = 2
    assert_raises(TypeError, bs2, alpha_1, alpha_2_w)
    alpha_1_w = 1.5
    assert_raises(ValueError, bs2, alpha_1_w, alpha_2)
    alpha_2_w = 2.5
    assert_raises(ValueError, bs2, alpha_1, alpha_2_w)

    # Checking output
    assert bs2(alpha_1, alpha_2) == 0.5
