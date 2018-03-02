"""Tests for flik.linesearch.linesearch."""

import numpy as np
from numpy.testing import assert_raises
from flik.linesearch.linesearch import line_search_general


def test_linesearch_general():
    """Test linesearch_general."""
    def fsq(var):
        r"""Little function :math:`\sum_i x_i^2`."""
        return np.sum(var**2)

    def grad(var):
        r"""Gradient the square function :math:`\sum_i x_i^2`."""
        return 2*var

    # Check conditions
    func = fsq
    var = np.array([0.2, 0.5])
    step = -grad(var)
    alpha = 0.1
    interpolation = 'bs1'
    assert_raises(TypeError, line_search_general, var, func, grad, step, alpha, 2,
                  interpolation)
    assert_raises(ValueError, line_search_general, var, func, grad, step, alpha,
                  'algo', interpolation)
    assert_raises(ValueError, line_search_general, var, func, grad, step, alpha,
                  ['algo', 'armijo'], interpolation)
    assert_raises(ValueError, line_search_general, var, func, grad, step, alpha,
                  ['armijo', 'armijo'], interpolation)
    # Check interpolation choice
    assert_raises(TypeError, line_search_general, var, func, grad, step, alpha,
                  ['armijo'], 2)
    assert_raises(ValueError, line_search_general, var, func, grad, step, alpha,
                  ['armijo'], 'algo')
    # Check constants
    assert_raises(TypeError, line_search_general, var, func, grad, step, alpha,
                  ['soft-wolfe'], interpolation, const1=2)
    assert_raises(TypeError, line_search_general, var, func, grad, step, alpha,
                  ['soft-wolfe'], interpolation, const1=0.1, const2=3)
    assert_raises(ValueError, line_search_general, var, func, grad, step, alpha,
                  ['soft-wolfe'], interpolation, const1=-0.1)
    assert_raises(ValueError, line_search_general, var, func, grad, step, alpha,
                  ['strong-wolfe'], interpolation, const1=2.1)
    assert_raises(ValueError, line_search_general, var, func, grad, step, alpha, ['armijo'],
                  interpolation, const1=0.1, const2=1.1)
    assert_raises(ValueError, line_search_general, var, func, grad, step, alpha, ['armijo'],
                  interpolation, const1=0.1, const2=-1.1)
    assert_raises(ValueError, line_search_general, var, func, grad, step, alpha, ['armijo'],
                  interpolation, const1=0.1, const2=0.01)
    # Check range of new alpha
    assert 0 < line_search_general(var, func, grad, step, alpha,
                                   'armijo', 'cubic') <= 1.0
    assert 0 < line_search_general(var, func, grad, step, alpha,
                                   ['armijo', 'strong-wolfe'], 'bs1') <= 1.0
    assert 0 < line_search_general(var, func, grad, step, alpha,
                                   'soft-wolfe', 'bs2') <= 1.0
