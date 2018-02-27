"""Test for flink.linesearch.input."""

import numpy as np
from nose.tools import assert_raises

from flik.linesearch.input import check_input

def test_input_control():
    """Tests for Line Search parameters"""
    def func0(var):
        r"""Function :math:`\sum_i x_i^2`"""
        return np.sum(var**2)

    def grad0(var):
        r"""Gradient of sum_square :math:`\nabla f(x)`"""
        return 2*var

    # Check var vector
    direction = np.array([2., 1., 3.])
    alpha = 0.1
    var = [1]
    assert_raises(TypeError, check_input, var, func0, grad0, direction, alpha)
    var = np.array([[1., 3.],[2., 1.]])
    assert_raises(TypeError, check_input, var, func0, grad0, direction, alpha)
    # Check for function and gradient
    var = np.array([2., 1., 3.])
    func1 = [0.1]
    assert_raises(TypeError, check_input, var, func1, grad0, direction, alpha)
    grad1 = 2.
    assert_raises(TypeError, check_input, var, func0, grad1, direction, alpha)
    # Check direction
    direction = [1, 2, 3]
    assert_raises(TypeError, check_input, var, func0, grad0, direction, alpha)
    direction = np.array([[1, 2],[5, 3]])
    assert_raises(TypeError, check_input, var, func0, grad0, direction, alpha)
    # Check alpha
    direction = np.array([2., 1., 3.])
    alpha = 2
    assert_raises(TypeError, check_input, var, func0, grad0, direction, alpha)
    alpha = 1.2
    assert_raises(ValueError, check_input, var, func0, grad0, direction, alpha)
    alpha = -1.2
    assert_raises(ValueError, check_input, var, func0, grad0, direction, alpha)
    assert check_input(var, func0, grad0, direction, 0.2) is None
