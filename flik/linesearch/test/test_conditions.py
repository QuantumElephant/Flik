"""Tests for flik.linesearch.conditions."""

import numpy as np
from numpy.testing import assert_raises
from flik.linesearch.conditions import wolfe, armijo


def test_wolfe():
    """Test the wolfe condition."""
    def grad(point):
        """Gradient of function :math:`point**2`."""
        return 2 * point

    # Set parameters
    point = np.array([1.8])
    step = np.array([-0.6])
    alpha = 0.8
    const2 = 0.9

    const2_w = 1
    assert_raises(TypeError, wolfe, grad, point, step, alpha, const2_w)
    const2_w = -0.209
    assert_raises(ValueError, wolfe, grad, point, step, alpha, const2_w)
    const2_w = 2.0
    assert_raises(ValueError, wolfe, grad, point, step, alpha, const2_w)

    # Making point and step float and int
    point_float = 1.8
    assert wolfe(grad, point_float, step, alpha, const2)
    point_int = 2
    assert wolfe(grad, point_int, step, alpha, const2)
    step_float = -0.6
    assert wolfe(grad, point, step_float, alpha, const2)
    step_int = -1
    assert wolfe(grad, point, step_int, alpha, const2)

    # Checking condition
    assert wolfe(grad, point, step, alpha, const2, strong_wolfe=True)
    assert not wolfe(grad, point, -step, alpha, const2, strong_wolfe=True)

    point = np.array([2., 1., 3.])
    step = np.array([2., 1., 3.])
    alpha = 0.1
    assert wolfe(grad, point, step, alpha, const2, strong_wolfe=False)
    assert not wolfe(grad, point, -step, alpha, const2, strong_wolfe=False)

    assert wolfe(grad, point, step, alpha, const2, strong_wolfe=False)
    assert not wolfe(grad, point, -step, alpha, const2, strong_wolfe=False)


def test_armijo():
    """Test Armijo condition."""
    def fsq(var):
        r"""Little function :math:`\sum_i x_i^2`."""
        return np.sum(var**2)

    def grad(var):
        r"""Gradient the square function :math:`\sum_i x_i^2`."""
        return 2*var

    # Check const1
    var = np.array([1.8])
    direction = grad(var)
    alpha = 0.5
    assert_raises(TypeError, armijo, func=fsq, grad=grad, point=var,
                  step=direction, alpha=alpha, const1=2)
    assert_raises(ValueError, armijo, func=fsq, grad=grad, point=var,
                  step=direction, alpha=alpha, const1=1.0)
    assert_raises(ValueError, armijo, func=fsq, grad=grad, point=var,
                  step=direction, alpha=alpha, const1=-1.0)

    # Check condition
    assert not armijo(fsq, grad, var, direction, alpha=alpha)
    assert armijo(fsq, grad, var, -direction, alpha=alpha)
