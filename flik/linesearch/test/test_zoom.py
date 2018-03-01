"""Tests for flik.linesearch.zoom."""

import numpy as np
from nose.tools import assert_raises

from flik.linesearch.zoom import zoom


def test_zoom():
    """Tests for the zoom function."""
    # Set parameters

    def func0(var):
        r"""Evaluate :math:`f(x)=x^T x`."""
        return var.dot(var)

    def grad0(var):
        r"""Evaluate gradient of sum_square :math:`\nabla f(x) = 2x`."""
        return 2*var

    var = np.array([1., 2., 0.5])
    direction = -grad0(var)
    func = func0
    grad = grad0
    alpha_lo = 0.1
    alpha_hi = 0.9
    const1 = 1e-2
    const2 = 1e-4
    # Check constants
    assert_raises(TypeError, zoom, var, func, grad, direction, 0.1, 0.9, const1=0, const2=0.9)
    assert_raises(TypeError, zoom, var, func, grad, direction, 0.1, 0.9, const1=0.1, const2=1)
    # Check constants' values
    assert_raises(ValueError, zoom, var, func, grad, direction, 0.1, 0.9, const1=0.1, const2=1.0)
    assert_raises(ValueError, zoom, var, func, grad, direction, 0.1, 0.9, const1=0.0, const2=0.9)
    assert_raises(ValueError, zoom, var, func, grad, direction, 0.1, 0.9, const1=0.9, const2=0.1)

    # Check zoom function range
    alpha_lo = 0.1
    alpha_hi = 0.9
    assert 0 < zoom(var, func, grad, direction, alpha_lo, alpha_hi) <= 1
